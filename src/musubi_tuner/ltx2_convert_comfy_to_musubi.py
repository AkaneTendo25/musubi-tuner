import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from safetensors import safe_open

from musubi_tuner.networks import lora as lora_module
from musubi_tuner.networks import lokr as lokr_module
from musubi_tuner.networks.lora_ltx2 import load_ltx2_transformer
from musubi_tuner.networks.lora_shared import _get_dense_weight_norm
from musubi_tuner.utils import model_utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_KNOWN_COMFY_SUFFIXES = (
    "lora_A.weight",
    "lora_B.weight",
    "alpha",
    "dora_scale",
    "initial_norm",
    "oft_R.weight",
    "oft_R.scaled_oft",
    "oft_block_size_metadata",
    "oft_block_share_metadata",
    "oft_coft_metadata",
    "coft_eps_metadata",
    "scaled_oft_metadata",
    "lokr_w1",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
)

_UNSUPPORTED_COMFY_SUFFIXES = (
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_t2",
)


def _build_lora_module_lookup(base_model):
    lookup = {}
    for name, module in base_model.named_modules():
        if not name:
            continue
        if module.__class__.__name__ not in {"Linear", "Conv2d"}:
            continue
        normalized = name.replace(".", "_")
        lookup[f"lora_unet_{normalized}"] = module
        lookup[f"lora_unet_model_{normalized}"] = module
    return lookup


def _convert_module_prefix_from_comfy(path: str) -> Optional[str]:
    if not path.startswith("diffusion_model."):
        return None

    path = path[len("diffusion_model.") :]
    if path.startswith("video_embeddings_connector."):
        path = path[len("video_embeddings_connector.") :]
        return f"lora_unet_embeddings_connector_{path.replace('.', '_')}"
    if path.startswith("audio_embeddings_connector."):
        path = path[len("audio_embeddings_connector.") :]
        return f"lora_unet_audio_embeddings_connector_{path.replace('.', '_')}"
    return f"lora_unet_model_{path.replace('.', '_')}"


def _convert_key_from_comfy_native(key: str) -> Optional[str]:
    for suffix in _UNSUPPORTED_COMFY_SUFFIXES:
        needle = f".{suffix}"
        if key.endswith(needle):
            raise ValueError(
                f"Unsupported native ComfyUI LoKr tensor '{suffix}' in key '{key}'. "
                "This standalone converter only supports the Linear LoKr layout used by LTX-2."
            )

    matched_suffix = None
    for suffix in _KNOWN_COMFY_SUFFIXES:
        needle = f".{suffix}"
        if key.endswith(needle):
            matched_suffix = suffix
            path = key[: -len(needle)]
            break
    if matched_suffix is None:
        return None

    module_name = _convert_module_prefix_from_comfy(path)
    if module_name is None:
        return None

    if matched_suffix == "lora_A.weight":
        weight_part = "lora_down.weight"
    elif matched_suffix == "lora_B.weight":
        weight_part = "lora_up.weight"
    else:
        weight_part = matched_suffix

    return f"{module_name}.{weight_part}"


def is_ltx2_comfy_adapter_state_dict(weights_sd: Dict[str, torch.Tensor]) -> bool:
    if not weights_sd:
        return False
    return any(
        key.startswith("diffusion_model.")
        and any(key.endswith(f".{suffix}") for suffix in _KNOWN_COMFY_SUFFIXES + _UNSUPPORTED_COMFY_SUFFIXES)
        for key in weights_sd.keys()
    )


def _group_converted_keys_by_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if "." not in key:
            continue
        module_name, suffix = key.split(".", 1)
        grouped.setdefault(module_name, {})[suffix] = tensor
    return grouped


def _module_state_is_lokr(module_state: Dict[str, torch.Tensor]) -> bool:
    return any(key in module_state for key in ("lokr_w1", "lokr_w1_a", "lokr_w2", "lokr_w2_a", "lokr_w2_b"))


def _get_comfy_base_weight_norm(base_weight: torch.Tensor) -> torch.Tensor:
    zero_delta = torch.zeros_like(base_weight)
    return _get_dense_weight_norm(base_weight, zero_delta, 0.0)


def _translate_comfy_dora_scale_to_musubi_magnitude(
    base_weight: torch.Tensor,
    dora_scale: torch.Tensor,
    weight_norm: torch.Tensor,
) -> torch.Tensor:
    base_norm = _get_comfy_base_weight_norm(base_weight)
    if base_norm.is_floating_point():
        eps = 1e-12 if base_norm.dtype in (torch.float32, torch.float64) else 1e-6
        base_norm = base_norm.clamp_min(eps)
    dora_scale = dora_scale.to(device=weight_norm.device, dtype=weight_norm.dtype).reshape(-1)
    weight_norm = weight_norm.reshape(-1)
    base_norm = base_norm.to(weight_norm.device, weight_norm.dtype).reshape(-1)
    magnitude = dora_scale * (weight_norm / base_norm)
    return magnitude.to(device=dora_scale.device, dtype=dora_scale.dtype)


def _resolve_lokr_scale(module_state: Dict[str, torch.Tensor]) -> float:
    if "lokr_w2" in module_state:
        return 1.0

    rank = int(module_state["lokr_w2_a"].shape[1])
    alpha = module_state.get("alpha", None)
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.item())
    elif alpha is None:
        alpha = float(rank)
    else:
        alpha = float(alpha)
    return alpha / float(rank)


def _reconstruct_musubi_magnitude(
    module_name: str,
    module_state: Dict[str, torch.Tensor],
    base_module_lookup: Dict[str, torch.nn.Module],
) -> torch.Tensor:
    base_module = base_module_lookup.get(module_name)
    if base_module is None or not hasattr(base_module, "weight"):
        raise KeyError(f"Could not resolve DoRA/DokR target module '{module_name}' in the provided base model")

    base_weight = lora_module._get_effective_module_weight(
        base_module,
        dtype=torch.float,
        detach=True,
    )
    dora_scale = module_state["dora_scale"]

    if "lokr_w1" in module_state:
        scale = _resolve_lokr_scale(module_state)
        diff_weight = lokr_module._materialize_lokr_weight_from_state_dict(module_state, scale, base_weight.device)
        diff_weight = diff_weight.to(device=base_weight.device, dtype=base_weight.dtype)
        weight_norm = lokr_module._get_dokr_weight_norm(base_weight, diff_weight)
        return _translate_comfy_dora_scale_to_musubi_magnitude(base_weight, dora_scale, weight_norm)

    if "lora_down.weight" not in module_state or "lora_up.weight" not in module_state:
        raise ValueError(f"Unsupported DoRA module '{module_name}': expected native LoRA A/B or native LoKr tensors")

    down_weight = module_state["lora_down.weight"].to(device=base_weight.device, dtype=torch.float)
    up_weight = module_state["lora_up.weight"].to(device=base_weight.device, dtype=torch.float)
    rank = int(down_weight.shape[0])
    alpha = module_state.get("alpha", None)
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.item())
    elif alpha is None:
        alpha = float(rank)
    else:
        alpha = float(alpha)
    scaling = alpha / float(rank)
    weight_norm = lora_module._get_dora_weight_norm(base_weight, down_weight, up_weight, scaling)
    return _translate_comfy_dora_scale_to_musubi_magnitude(base_weight, dora_scale, weight_norm)


def _ensure_alpha_defaults(module_state: Dict[str, torch.Tensor]) -> None:
    if "alpha" in module_state:
        return

    if "lora_down.weight" in module_state:
        module_state["alpha"] = torch.tensor(int(module_state["lora_down.weight"].shape[0]), dtype=torch.float32)
        return

    if "lokr_w2_a" in module_state:
        module_state["alpha"] = torch.tensor(int(module_state["lokr_w2_a"].shape[1]), dtype=torch.float32)
        return

    if "lokr_w2" in module_state:
        module_state["alpha"] = torch.tensor(int(max(module_state["lokr_w2"].shape)), dtype=torch.float32)


def _infer_lokr_factor(module_state: Dict[str, torch.Tensor]) -> Optional[int]:
    if "lokr_w1" not in module_state:
        return None

    if "lokr_w2" in module_state:
        out_k, in_n = module_state["lokr_w2"].shape
    elif "lokr_w2_a" in module_state and "lokr_w2_b" in module_state:
        out_k = int(module_state["lokr_w2_a"].shape[0])
        in_n = int(module_state["lokr_w2_b"].shape[1])
    else:
        return None

    out_l, in_m = module_state["lokr_w1"].shape
    in_dim = int(in_m) * int(in_n)
    out_dim = int(out_l) * int(out_k)
    target_in = (int(in_m), int(in_n))
    target_out = (int(out_l), int(out_k))

    if lokr_module.factorization(in_dim, -1) == target_in and lokr_module.factorization(out_dim, -1) == target_out:
        return -1

    max_factor = max(in_dim, out_dim)
    for factor in range(1, max_factor + 1):
        if lokr_module.factorization(in_dim, factor) == target_in and lokr_module.factorization(out_dim, factor) == target_out:
            return factor
    return None


def _infer_lokr_network_args(grouped_modules: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, str]:
    factors = {
        inferred
        for module_state in grouped_modules.values()
        for inferred in [_infer_lokr_factor(module_state)]
        if inferred is not None
    }
    if len(factors) > 1:
        raise ValueError(f"Could not infer a single global LoKr factor for this checkpoint: {sorted(factors)}")
    if not factors:
        return {}
    factor = next(iter(factors))
    if factor == -1:
        return {}
    return {"factor": str(factor)}


def _load_network_args_metadata(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid ss_network_args JSON in input metadata")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignoring non-dict ss_network_args metadata in input file")
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _infer_network_metadata(
    grouped_modules: Dict[str, Dict[str, torch.Tensor]],
    existing_metadata: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    existing_metadata = existing_metadata or {}
    network_args = _load_network_args_metadata(existing_metadata.get("ss_network_args"))

    has_lokr = any(any(key.startswith("lokr_") for key in module_state) for module_state in grouped_modules.values())
    has_lora = any(
        "lora_down.weight" in module_state or "lora_up.weight" in module_state for module_state in grouped_modules.values()
    )
    has_dora = any("lora_magnitude_vector.weight" in module_state for module_state in grouped_modules.values())
    has_oft = any("oft_R.weight" in module_state for module_state in grouped_modules.values())
    has_dora_oft = any("dora_scale" in module_state for module_state in grouped_modules.values()) and has_oft

    if has_lokr and has_lora:
        raise ValueError("Mixed native LoRA and native LoKr tensors in a single LTX-2 checkpoint are not supported")

    if has_lokr:
        network_module = "networks.lokr"
        if has_oft:
            network_args["use_dora_oft"] = "true"
            network_args.pop("use_dora", None)
        elif has_dora:
            network_args["use_dora"] = "true"
            network_args.pop("use_dora_oft", None)
        else:
            network_args.pop("use_dora", None)
            network_args.pop("use_dora_oft", None)
        network_args.update(_infer_lokr_network_args(grouped_modules))
        return network_module, network_args

    network_module = "networks.lora_ltx2"
    if has_dora_oft:
        network_args["use_dora_oft"] = "true"
        network_args.pop("use_dora", None)
    elif has_oft:
        network_args.pop("use_dora", None)
        network_args.pop("use_dora_oft", None)
        network_args.pop("use_oft", None)
    elif has_dora:
        network_args["use_dora"] = "true"
        network_args.pop("use_dora_oft", None)
    else:
        network_args.pop("use_dora", None)
        network_args.pop("use_dora_oft", None)
    network_args.pop("factor", None)
    return network_module, network_args


def _convert_comfy_state_dict_to_intermediate(comfy_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted_intermediate = {}
    for key, tensor in comfy_state_dict.items():
        new_key = _convert_key_from_comfy_native(key)
        if new_key is None:
            continue
        converted_intermediate[new_key] = tensor
    return converted_intermediate


def _grouped_modules_need_base_model(grouped_modules: Dict[str, Dict[str, torch.Tensor]]) -> bool:
    return any(
        "dora_scale" in module_state and ("oft_R.weight" not in module_state or _module_state_is_lokr(module_state))
        for module_state in grouped_modules.values()
    )


def _comfy_state_dict_needs_base_model(comfy_state_dict: Dict[str, torch.Tensor]) -> bool:
    converted_intermediate = _convert_comfy_state_dict_to_intermediate(comfy_state_dict)
    grouped_modules = _group_converted_keys_by_module(converted_intermediate)
    return _grouped_modules_need_base_model(grouped_modules)


def convert_ltx2_comfy_to_musubi_state_dict(
    comfy_state_dict: Dict[str, torch.Tensor],
    *,
    base_model=None,
    existing_metadata: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, str]]:
    if not is_ltx2_comfy_adapter_state_dict(comfy_state_dict):
        raise ValueError("Input state dict does not look like a native LTX-2 ComfyUI adapter")

    converted_intermediate = _convert_comfy_state_dict_to_intermediate(comfy_state_dict)

    grouped_modules = _group_converted_keys_by_module(converted_intermediate)
    needs_base_model = _grouped_modules_need_base_model(grouped_modules)
    base_lookup = None
    if needs_base_model:
        if base_model is None:
            raise ValueError(
                "Exact DoRA/DokR reconstruction requires the original LTX-2 base transformer. "
                "Pass --base_model/--dit to this standalone converter."
            )
        base_lookup = _build_lora_module_lookup(base_model)

    converted_state_dict = {}
    for module_name, module_state in grouped_modules.items():
        module_state = dict(module_state)
        if "dora_scale" in module_state and ("oft_R.weight" not in module_state or _module_state_is_lokr(module_state)):
            magnitude = _reconstruct_musubi_magnitude(module_name, module_state, base_lookup)
            module_state["lora_magnitude_vector.weight"] = magnitude
            module_state.pop("dora_scale", None)
        _ensure_alpha_defaults(module_state)

        for suffix, tensor in module_state.items():
            converted_state_dict[f"{module_name}.{suffix}"] = tensor

    network_module, network_args = _infer_network_metadata(
        grouped_modules=_group_converted_keys_by_module(converted_state_dict),
        existing_metadata=existing_metadata,
    )
    return converted_state_dict, network_module, network_args


def convert_ltx2_comfy_to_musubi(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    base_model_path: Optional[str] = None,
    audio_video: bool = False,
    base_dtype: str = "float32",
    device: str = "cpu",
) -> Path:
    logger.info("Loading ComfyUI adapter from %s", input_path)
    comfy_state_dict = safetensors.torch.load_file(input_path)

    metadata = {}
    with safe_open(input_path, framework="pt") as handle:
        metadata = dict(handle.metadata() or {})

    needs_base_model = _comfy_state_dict_needs_base_model(comfy_state_dict)
    base_model = None
    if needs_base_model:
        if not base_model_path:
            raise ValueError(
                "This checkpoint uses native DoRA/DokR tensors (.dora_scale) without preserved OFT rotation state. "
                "Pass --base_model or --dit with the original LTX-2 transformer path for exact reconstruction."
            )
        logger.info("Loading base LTX-2 transformer from %s", base_model_path)
        base_model = load_ltx2_transformer(
            model_path=base_model_path,
            device=torch.device(device),
            dtype=model_utils.str_to_dtype(base_dtype, torch.float32),
            audio_video=audio_video,
        )

    converted_state_dict, network_module, network_args = convert_ltx2_comfy_to_musubi_state_dict(
        comfy_state_dict,
        base_model=base_model,
        existing_metadata=metadata,
    )

    output_metadata = dict(metadata)
    output_metadata["ss_network_module"] = network_module
    if network_args:
        output_metadata["ss_network_args"] = json.dumps(network_args)
    else:
        output_metadata.pop("ss_network_args", None)
    output_metadata = {str(k): str(v) for k, v in output_metadata.items()}

    model_utils.precalculate_safetensors_hashes(converted_state_dict, output_metadata)

    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}.musubi{input_file.suffix}"
    else:
        output_path = Path(output_path)

    logger.info("Saving Musubi adapter to %s", output_path)
    safetensors.torch.save_file(converted_state_dict, str(output_path), metadata=output_metadata)
    logger.info("Saved Musubi adapter with ss_network_module=%s", network_module)
    if network_args:
        logger.info("Saved Musubi adapter with ss_network_args=%s", json.dumps(network_args))
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a native LTX-2 ComfyUI adapter back to native Musubi format")
    parser.add_argument("--input", type=str, required=True, help="Path to the input ComfyUI adapter safetensors file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the converted Musubi adapter")
    parser.add_argument(
        "--base_model",
        "--dit",
        dest="base_model",
        type=str,
        default=None,
        help=(
            "Path to the original LTX-2 base transformer. Required only when reconstructing "
            "DoRA/DokR magnitude from Comfy dora_scale without preserved OFT rotation state."
        ),
    )
    parser.add_argument(
        "--audio_video",
        action="store_true",
        help="Load the audio-video LTX-2 transformer variant when resolving base weights.",
    )
    parser.add_argument(
        "--base_dtype",
        type=str,
        default="float32",
        help="Dtype used when loading the base transformer (default: float32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used when loading the base transformer (default: cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_ltx2_comfy_to_musubi(
        input_path=args.input,
        output_path=args.output,
        base_model_path=args.base_model,
        audio_video=args.audio_video,
        base_dtype=args.base_dtype,
        device=args.device,
    )


if __name__ == "__main__":
    main()
