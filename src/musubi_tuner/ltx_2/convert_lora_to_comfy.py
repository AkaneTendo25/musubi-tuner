"""
Convert LTX-2 adapter weights from training format to ComfyUI format.

Training format:
  - Keys: lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
  - Uses underscores as separators
  - Uses native Musubi DoRA magnitude keys: .lora_magnitude_vector.weight

ComfyUI format:
  - Keys: diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
  - Uses dots as separators
  - Preserves adapter-native keys such as .alpha, .lokr_w1, .lokr_w2_a, etc.
  - Uses .dora_scale for ComfyUI's DoRA-aware loaders
  - Preserves Musubi DoRA-OFT / DoKr-OFT keys as .oft_R.*. These are native
    Musubi/patched-loader tensors, not stock ComfyUI OFT .oft_blocks tensors.
"""

import safetensors.torch
import argparse
import gc
import os
from pathlib import Path
import torch

from musubi_tuner.networks import lora as lora_module
from musubi_tuner.networks import lokr as lokr_module
from musubi_tuner.utils import model_utils


def _release_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _group_lora_keys_by_module(state_dict):
    grouped = {}
    for key, tensor in state_dict.items():
        if "." not in key:
            continue
        module_name, suffix = key.split(".", 1)
        grouped.setdefault(module_name, {})[suffix] = tensor
    return grouped


def _build_lora_module_lookup(base_model):
    lookup = {}
    for name, module in base_model.named_modules():
        if not name:
            continue
        if module.__class__.__name__ not in {"Linear", "Conv2d"}:
            continue
        lora_name = f"lora_unet_{name.replace('.', '_')}"
        lookup[lora_name] = module
        # Training-time keys may include the wrapper's `.model` prefix while
        # standalone converters often receive the bare LTXModel.
        lookup[f"lora_unet_model_{name.replace('.', '_')}"] = module
        if name.startswith("model."):
            lookup[f"lora_unet_{name[len('model.'):].replace('.', '_')}"] = module
    return lookup


def _resolve_lokr_scale(module_state: dict[str, torch.Tensor]) -> float:
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


def _get_comfy_base_weight_norm(base_weight: torch.Tensor) -> torch.Tensor:
    zero_delta = torch.zeros_like(base_weight)
    return lora_module._get_dense_weight_norm(base_weight, zero_delta, 0.0)


def _translate_musubi_magnitude_to_comfy_dora_scale(
    base_weight: torch.Tensor,
    magnitude: torch.Tensor,
    weight_norm: torch.Tensor,
) -> torch.Tensor:
    base_norm = _get_comfy_base_weight_norm(base_weight)
    if weight_norm.is_floating_point():
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
    dora_scale = magnitude.to(device=base_norm.device, dtype=base_norm.dtype) * (base_norm / weight_norm)
    if base_weight.dim() >= 2 and dora_scale.dim() == 1 and int(dora_scale.shape[0]) == int(base_weight.shape[0]):
        return dora_scale.view(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))
    return dora_scale


def _convert_dora_module_to_comfy_scale(
    module_state: dict[str, torch.Tensor],
    base_module: torch.nn.Module,
) -> torch.Tensor:
    base_weight = lora_module._get_effective_module_weight(base_module, dtype=torch.float, detach=True)
    magnitude = module_state["lora_magnitude_vector.weight"]

    if "lokr_w1" in module_state:
        scale = _resolve_lokr_scale(module_state)
        diff_weight = lokr_module._materialize_lokr_weight_from_state_dict(module_state, scale, base_weight.device)
        diff_weight = diff_weight.to(device=base_weight.device, dtype=base_weight.dtype)
        weight_norm = lokr_module._get_dokr_weight_norm(base_weight, diff_weight)
        return _translate_musubi_magnitude_to_comfy_dora_scale(base_weight, magnitude, weight_norm)

    if "lora_down.weight" not in module_state or "lora_up.weight" not in module_state:
        raise ValueError("Unsupported DoRA module state: expected LoRA or LoKr adapter tensors")

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
    return _translate_musubi_magnitude_to_comfy_dora_scale(base_weight, magnitude, weight_norm)


def _module_state_is_lokr(module_state: dict[str, torch.Tensor]) -> bool:
    return any(key in module_state for key in ("lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b"))


def _module_state_is_oft(module_state: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("oft_") or key.startswith("oft_R.") for key in module_state.keys())


def _state_dict_has_native_oft_adapter(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".oft_R.weight" in key for key in state_dict.keys())


def convert_key_to_comfy(key):
    """
    Convert a training format key to ComfyUI format

    Example:
        lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
        -> diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
    """
    # Split into main part and weight part
    parts = key.split('.')
    if len(parts) < 2:
        print(f"Warning: Unexpected key format: {key}")
        return None

    main_part = parts[0]  # e.g., lora_unet_model_transformer_blocks_0_attn1_to_k
    weight_part = '.'.join(parts[1:])  # e.g., lora_down.weight

    # Remove the lora_unet_ prefix and handle the wrapper's module structure.
    # Transformer keys: lora_unet_model_transformer_blocks_... (wrapper.model.transformer_blocks)
    # Connector keys:   lora_unet_embeddings_connector_... (wrapper.embeddings_connector)
    if main_part.startswith('lora_unet_model_'):
        # Standard transformer path: strip wrapper.model prefix
        main_part = main_part[len('lora_unet_model_'):]
    elif main_part.startswith('lora_unet_'):
        # Connector or other wrapper-level module
        main_part = main_part[len('lora_unet_'):]
    else:
        print(f"Warning: Key doesn't start with 'lora_unet_': {key}")
        return None

    # Map connector attribute names to ComfyUI model names
    # Training wrapper: self.embeddings_connector -> ComfyUI: video_embeddings_connector
    if main_part.startswith('embeddings_connector_'):
        main_part = 'video_' + main_part
    # audio_embeddings_connector is already correct

    # Convert underscores to dots for the hierarchy
    # We need to be careful with numeric parts
    # transformer_blocks_0_attn1_to_k -> transformer_blocks.0.attn1.to_k

    # Strategy: Replace patterns carefully
    # Replace _NUMBER_ with .NUMBER.
    # Replace other underscores based on context

    converted = main_part

    # IMPORTANT ORDER: Handle audio patterns BEFORE general attn patterns!
    # This prevents _attn1_ from matching inside audio_attn1_
    import re

    # Step 0: Handle connector module paths
    # video_embeddings_connector_transformer_1d_blocks_0_... -> video_embeddings_connector.transformer_1d_blocks.0....
    # audio_embeddings_connector_transformer_1d_blocks_0_... -> audio_embeddings_connector.transformer_1d_blocks.0....
    converted = converted.replace('video_embeddings_connector_', 'video_embeddings_connector.')
    converted = converted.replace('audio_embeddings_connector_', 'audio_embeddings_connector.')
    converted = converted.replace('transformer_1d_blocks_', 'transformer_1d_blocks.')

    # Step 1: Basic block structure (main transformer)
    converted = converted.replace('transformer_blocks_', 'transformer_blocks.')

    # Step 2: Handle audio/video attention patterns FIRST (keep underscores in these)
    converted = converted.replace('_audio_attn1_', '.audio_attn1.')
    converted = converted.replace('_audio_attn2_', '.audio_attn2.')
    converted = converted.replace('_audio_to_video_attn_', '.audio_to_video_attn.')
    converted = converted.replace('_video_to_audio_attn_', '.video_to_audio_attn.')
    converted = converted.replace('_audio_ff_', '.audio_ff.')

    # Step 3: Now handle regular (non-audio) attention patterns
    converted = converted.replace('_attn1_', '.attn1.')
    converted = converted.replace('_attn2_', '.attn2.')

    # Step 4: Handle projection layers (use regex to avoid matching inside audio_to_video, video_to_audio)
    # Only match _to_k/_to_q/_to_v at word boundaries (end of string or followed by .)
    converted = re.sub(r'_to_k($|\.)', r'.to_k\1', converted)
    converted = re.sub(r'_to_q($|\.)', r'.to_q\1', converted)
    converted = re.sub(r'_to_v($|\.)', r'.to_v\1', converted)
    converted = re.sub(r'_to_gate_logits($|\.)', r'.to_gate_logits\1', converted)
    converted = re.sub(r'_to_out\.', r'.to_out.', converted)
    converted = re.sub(r'_to_out_', r'.to_out.', converted)

    # Step 5: Handle feedforward layers
    converted = converted.replace('_ff_net_', '.ff.net.')
    converted = converted.replace('_ff_', '.ff.')
    converted = converted.replace('_net_', '.net.')
    converted = converted.replace('_proj', '.proj')

    # Step 6: Fix remaining numeric suffixes
    converted = re.sub(r'to_out_(\d+)', r'to_out.\1', converted)
    converted = re.sub(r'net_(\d+)', r'net.\1', converted)

    # Build the final key
    comfy_key = f"diffusion_model.{converted}"

    # Convert weight naming for ComfyUI adapter loaders.
    if 'lora_down' in weight_part:
        weight_part = weight_part.replace('lora_down', 'lora_A')
    elif 'lora_up' in weight_part:
        weight_part = weight_part.replace('lora_up', 'lora_B')
    elif weight_part == 'lora_magnitude_vector.weight':
        weight_part = 'dora_scale'

    comfy_key = f"{comfy_key}.{weight_part}"

    return comfy_key


def convert_key_from_comfy(key):
    """
    Convert a ComfyUI-format LTX-2 LoRA key back to training format.

    Example:
        diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
        -> lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
    """
    if key.endswith(".lora_A.weight"):
        weight_part = "lora_down.weight"
        path = key[: -len(".lora_A.weight")]
    elif key.endswith(".lora_B.weight"):
        weight_part = "lora_up.weight"
        path = key[: -len(".lora_B.weight")]
    elif key.endswith(".alpha"):
        weight_part = "alpha"
        path = key[: -len(".alpha")]
    elif key.endswith(".dora_scale"):
        weight_part = "dora_scale"
        path = key[: -len(".dora_scale")]
    elif key.endswith(".initial_norm"):
        weight_part = "initial_norm"
        path = key[: -len(".initial_norm")]
    elif key.endswith(".oft_R.weight"):
        weight_part = "oft_R.weight"
        path = key[: -len(".oft_R.weight")]
    elif key.endswith(".oft_R.scaled_oft"):
        weight_part = "oft_R.scaled_oft"
        path = key[: -len(".oft_R.scaled_oft")]
    elif key.endswith(".oft_block_size_metadata"):
        weight_part = "oft_block_size_metadata"
        path = key[: -len(".oft_block_size_metadata")]
    elif key.endswith(".oft_block_share_metadata"):
        weight_part = "oft_block_share_metadata"
        path = key[: -len(".oft_block_share_metadata")]
    elif key.endswith(".oft_coft_metadata"):
        weight_part = "oft_coft_metadata"
        path = key[: -len(".oft_coft_metadata")]
    elif key.endswith(".coft_eps_metadata"):
        weight_part = "coft_eps_metadata"
        path = key[: -len(".coft_eps_metadata")]
    elif key.endswith(".scaled_oft_metadata"):
        weight_part = "scaled_oft_metadata"
        path = key[: -len(".scaled_oft_metadata")]
    else:
        return None

    if not path.startswith("diffusion_model."):
        return None

    path = path[len("diffusion_model.") :]
    if path.startswith("video_embeddings_connector."):
        path = path[len("video_embeddings_connector.") :]
        main_part = f"lora_unet_embeddings_connector_{path.replace('.', '_')}"
    elif path.startswith("audio_embeddings_connector."):
        path = path[len("audio_embeddings_connector.") :]
        main_part = f"lora_unet_audio_embeddings_connector_{path.replace('.', '_')}"
    else:
        main_part = f"lora_unet_model_{path.replace('.', '_')}"

    return f"{main_part}.{weight_part}"


def is_comfy_lora_state_dict(weights_sd):
    """Return True if the state dict looks like an LTX-2 ComfyUI adapter."""
    if not weights_sd:
        return False
    recognized_suffixes = (
        ".lora_A.weight",
        ".lora_B.weight",
        ".alpha",
        ".dora_scale",
        ".initial_norm",
        ".oft_R.weight",
        ".oft_R.scaled_oft",
        ".oft_block_size_metadata",
        ".oft_block_share_metadata",
        ".oft_coft_metadata",
        ".coft_eps_metadata",
        ".scaled_oft_metadata",
        ".lokr_w1",
        ".lokr_w2",
        ".lokr_w2_a",
        ".lokr_w2_b",
    )
    return any(
        key.startswith("diffusion_model.") and key.endswith(recognized_suffixes)
        for key in weights_sd.keys()
    )


def _is_ltx2_gate_logits_module(module_name: str) -> bool:
    return module_name.endswith("_to_gate_logits")


def _is_ltx2_feed_forward_module(module_name: str) -> bool:
    return "_ff_" in module_name


def convert_lora_to_comfy_state_dict(
    trained_state_dict,
    verbose=False,
    base_model=None,
    skip_gate_logits_dora=False,
    dora_ff_only=False,
):
    """Convert a training-format LTX-2 LoRA state dict to ComfyUI format."""
    converted_state_dict = dict(trained_state_dict)
    grouped_modules = _group_lora_keys_by_module(converted_state_dict)

    # ComfyUI uses adapter-native keys for LoKr and alpha, but DoRA variants must
    # be translated from Musubi's raw magnitude vector into ComfyUI's dora_scale.
    dora_modules = [
        module_name
        for module_name, module_state in grouped_modules.items()
        if "lora_magnitude_vector.weight" in module_state
    ]
    module_lookup = None
    if dora_modules:
        if base_model is None:
            raise ValueError(
                "DoRA and DokR LTX-2 conversion to ComfyUI requires the live base transformer. "
                "Training-time checkpoint export provides this automatically; standalone DoRA/DokR conversion is unsupported "
                "without base_model."
            )
        module_lookup = _build_lora_module_lookup(base_model)

    for module_name in dora_modules:
        module_state = grouped_modules[module_name]
        is_stock_comfy_dokr = _module_state_is_lokr(module_state) and not _module_state_is_oft(module_state)

        if dora_ff_only and not _is_ltx2_feed_forward_module(module_name):
            converted_state_dict.pop(f"{module_name}.lora_magnitude_vector.weight", None)
            if is_stock_comfy_dokr:
                converted_state_dict.pop(f"{module_name}.initial_norm", None)
            if verbose:
                print(f"Skipped ComfyUI DoRA scale for non-FF LTX-2 module: {module_name}")
            continue

        if skip_gate_logits_dora and _is_ltx2_gate_logits_module(module_name):
            converted_state_dict.pop(f"{module_name}.lora_magnitude_vector.weight", None)
            if is_stock_comfy_dokr:
                converted_state_dict.pop(f"{module_name}.initial_norm", None)
            if verbose:
                print(f"Skipped ComfyUI DoRA scale for LTX-2 gate logits: {module_name}")
            continue

        base_module = module_lookup.get(module_name)
        if base_module is None or not hasattr(base_module, "weight"):
            raise KeyError(f"Could not resolve DoRA target module '{module_name}' in the provided base model")

        comfy_scale = _convert_dora_module_to_comfy_scale(module_state, base_module)
        converted_state_dict.pop(f"{module_name}.lora_magnitude_vector.weight", None)
        if is_stock_comfy_dokr:
            converted_state_dict.pop(f"{module_name}.initial_norm", None)
        converted_state_dict[f"{module_name}.dora_scale"] = comfy_scale
        if verbose:
            print(
                f"Translated Musubi DoRA magnitude to ComfyUI dora_scale: {module_name}"
            )

    # Convert keys
    comfy_state_dict = {}
    converted = 0
    failed = 0

    for key, tensor in converted_state_dict.items():
        new_key = convert_key_to_comfy(key)

        if new_key is None:
            failed += 1
            print(f"Failed to convert key: {key}")
        else:
            comfy_state_dict[new_key] = tensor
            converted += 1
            if verbose:
                print(f"Converted: {key} -> {new_key}")

    if verbose:
        print("\nConversion summary:")
        print(f"  Converted: {converted} keys")
        print(f"  Failed: {failed} keys")
        print(f"  Output LoRA has {len(comfy_state_dict)} keys")

    return comfy_state_dict


def convert_lora_from_comfy_state_dict(comfy_state_dict):
    """
    Convert a ComfyUI-format LTX-2 adapter state dict back to training format.

    Notes:
        - This reverse helper only handles the plain LoRA A/B layout.
        - Native ComfyUI DoRA/LoKr/DokR exports are preserved correctly on
          Musubi -> ComfyUI export, but they are not reconstructed here as
          native Musubi DoRA/LoKr/DokR checkpoints.
    """
    converted_state_dict = {}
    lora_dims = {}

    for key, tensor in comfy_state_dict.items():
        new_key = convert_key_from_comfy(key)
        if new_key is None:
            continue
        converted_state_dict[new_key] = tensor
        if new_key.endswith(".lora_down.weight"):
            lora_name = new_key.rsplit(".", 2)[0]
            lora_dims[lora_name] = tensor.shape[0]

    for lora_name, dim in lora_dims.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in converted_state_dict:
            converted_state_dict[alpha_key] = torch.tensor(dim)

    return converted_state_dict


def convert_lora_to_comfy(
    input_path,
    output_path=None,
    verbose=False,
    base_model=None,
    base_model_path=None,
    audio_video=False,
    audio_only_model=False,
    base_dtype="float32",
    device="cpu",
    fp8_base=False,
    fp8_scaled=False,
    fp8_w8a8=False,
    w8a8_mode="int8",
    fp8_keep_blocks=None,
    nf4_base=False,
    nf4_block_size=32,
    quantize_device=None,
    skip_gate_logits_dora=False,
    dora_ff_only=False,
):
    """
    Convert a LoRA file from training format to ComfyUI format

    Args:
        input_path: Path to the input LoRA file
        output_path: Path to save the converted LoRA (optional)
        verbose: Print detailed conversion info
        base_model: Live base transformer module, required for exact DokR export

    Returns:
        Path to the output file
    """
    print(f"Loading LoRA from: {input_path}")
    loaded_base_model = False
    trained_state_dict = None
    comfy_state_dict = None

    try:
        # Load the trained LoRA
        trained_state_dict = safetensors.torch.load_file(input_path)

        print(f"Input LoRA has {len(trained_state_dict)} keys")

        if base_model is None and any(".lora_magnitude_vector.weight" in key for key in trained_state_dict.keys()):
            if base_model_path is not None:
                from musubi_tuner.ltx2_model_loading import load_ltx2_model

                print(f"Loading base LTX-2 transformer from: {base_model_path}")
                if nf4_base and fp8_base:
                    raise ValueError("nf4_base and fp8_base are mutually exclusive")
                if fp8_scaled and not fp8_base:
                    raise ValueError("fp8_scaled requires fp8_base")
                if fp8_w8a8 and not fp8_scaled:
                    raise ValueError("fp8_w8a8 requires fp8_scaled")

                base_torch_dtype = model_utils.str_to_dtype(base_dtype, torch.float32)
                weight_dtype = torch.float8_e4m3fn if fp8_base and not fp8_scaled else base_torch_dtype
                base_model = load_ltx2_model(
                    model_path=base_model_path,
                    device=torch.device(device),
                    load_device=torch.device(device),
                    torch_dtype=weight_dtype,
                    audio_video=audio_video,
                    audio_only_model=audio_only_model,
                    fp8_scaled=fp8_scaled,
                    fp8_w8a8=fp8_w8a8,
                    w8a8_mode=w8a8_mode,
                    fp8_keep_blocks=fp8_keep_blocks,
                    nf4_base=nf4_base,
                    nf4_block_size=nf4_block_size,
                    quantize_device=quantize_device,
                    lora_rank=0,
                    attn_mode="torch",
                    ffn_chunk_target=None,
                    ffn_chunk_size=0,
                    split_attn_target=None,
                    split_attn_mode=None,
                    split_attn_chunk_size=0,
                )
                loaded_base_model = True

        comfy_state_dict = convert_lora_to_comfy_state_dict(
            trained_state_dict,
            verbose=verbose,
            base_model=base_model,
            skip_gate_logits_dora=skip_gate_logits_dora,
            dora_ff_only=dora_ff_only,
        )

        print(f"Output LoRA has {len(comfy_state_dict)} keys")
        if _state_dict_has_native_oft_adapter(comfy_state_dict):
            print(
                "Note: this export preserves native Musubi DoRA-OFT/DoKr-OFT tensors "
                "(.oft_R.*). Stock ComfyUI's OFT loader expects .oft_blocks and will not "
                "apply these OFT rotations without a patched/custom loader."
            )

        # Determine output path
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.comfy{input_file.suffix}"

        # Load metadata from the original file
        metadata = None
        try:
            with safetensors.safe_open(input_path, framework="pt") as f:
                metadata = f.metadata()
            if metadata:
                print(f"Preserving {len(metadata)} metadata entries")
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")

        # Save the converted LoRA
        print(f"\nSaving LTX-2 Comfy-format LoRA to: {output_path}")
        comfy_state_dict = {
            key: tensor.to(dtype=torch.float32) if tensor.is_floating_point() else tensor
            for key, tensor in comfy_state_dict.items()
        }
        safetensors.torch.save_file(comfy_state_dict, output_path, metadata=metadata)

        print("[OK] Conversion complete!")

        return output_path
    finally:
        trained_state_dict = None
        comfy_state_dict = None
        if loaded_base_model:
            base_model = None
        _release_torch_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-2 LoRA from training format to ComfyUI format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input LoRA file (training format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the converted LoRA (default: <input>.comfy.safetensors)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )
    parser.add_argument(
        "--skip_gate_logits_dora",
        action="store_true",
        help="Do not export DoRA dora_scale for LTX-2 to_gate_logits modules. Keeps the LoKr weights for those modules.",
    )
    parser.add_argument(
        "--dora_ff_only",
        action="store_true",
        help="Export ComfyUI DoRA dora_scale only for LTX-2 feed-forward modules. Keeps LoKr weights everywhere.",
    )
    parser.add_argument(
        "--base_model",
        "--dit",
        dest="base_model",
        type=str,
        default=None,
        help="Path to the original LTX-2 base transformer. Required for standalone DoRA/DokR conversion.",
    )
    parser.add_argument(
        "--audio_video",
        action="store_true",
        help="Load the audio-video LTX-2 transformer variant when resolving base weights.",
    )
    parser.add_argument(
        "--audio_only_model",
        action="store_true",
        help="Load the audio-only LTX-2 transformer variant when resolving base weights.",
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
    parser.add_argument(
        "--fp8_base",
        action="store_true",
        help="Load base transformer weights using the same non-scaled FP8 mode as training.",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="Load base transformer weights using scaled FP8 quantization.",
    )
    parser.add_argument(
        "--fp8_w8a8",
        action="store_true",
        help="Apply W8A8 activation-quantization patch after scaled FP8 loading.",
    )
    parser.add_argument(
        "--w8a8_mode",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
        help="W8A8 quantization format (default: int8).",
    )
    parser.add_argument(
        "--fp8_keep_blocks",
        type=str,
        default=None,
        help="Transformer block indices to keep in high precision when --fp8_scaled is enabled.",
    )
    parser.add_argument(
        "--nf4_base",
        action="store_true",
        help="Load base transformer weights using NF4 quantization.",
    )
    parser.add_argument(
        "--nf4_block_size",
        type=int,
        default=32,
        help="NF4 block size (default: 32).",
    )
    parser.add_argument(
        "--quantize_device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "gpu"],
        help="Device for FP8/NF4 quantization math.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    try:
        output_path = convert_lora_to_comfy(
            args.input,
            args.output,
            args.verbose,
            base_model_path=args.base_model,
            audio_video=args.audio_video,
            audio_only_model=args.audio_only_model,
            base_dtype=args.base_dtype,
            device=args.device,
            fp8_base=args.fp8_base,
            fp8_scaled=args.fp8_scaled,
            fp8_w8a8=args.fp8_w8a8,
            w8a8_mode=args.w8a8_mode,
            fp8_keep_blocks=args.fp8_keep_blocks,
            nf4_base=args.nf4_base,
            nf4_block_size=args.nf4_block_size,
            quantize_device=args.quantize_device,
            skip_gate_logits_dora=args.skip_gate_logits_dora,
            dora_ff_only=args.dora_ff_only,
        )
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
