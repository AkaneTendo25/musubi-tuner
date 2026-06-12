"""
Convert LTX-2 LoRA from training format to ComfyUI format.

Compatibility module for existing training-time imports. New standalone
one-shot workflows should prefer ``python -m musubi_tuner.ltx2_convert_lora_to_comfy``.

Training format:
  - Keys: lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
  - Uses underscores as separators
  - Has separate .alpha keys
  - Uses native DoRA magnitude keys: .lora_magnitude_vector.weight

ComfyUI format:
  - Keys: diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
  - Uses dots as separators
  - Renames lora_down -> lora_A, lora_up -> lora_B
  - Folds alpha into lora_B for standard LoRA
  - Uses .dora_scale for ComfyUI's DoRA-aware loaders
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


def _build_lora_base_module_lookup(network):
    if network is None:
        return {}

    lookup = {}
    for attr_name in ("text_encoder_loras", "unet_loras"):
        for module in getattr(network, attr_name, []) or []:
            lora_name = getattr(module, "lora_name", None)
            org_module_ref = getattr(module, "org_module_ref", None)
            if not lora_name or not org_module_ref:
                continue
            lookup[lora_name] = org_module_ref[0]
    return lookup


def _build_lora_module_lookup(base_model):
    if base_model is None:
        return {}

    lookup = {}
    for name, module in base_model.named_modules():
        if not name:
            continue
        if module.__class__.__name__ not in {"Linear", "Conv2d"}:
            continue
        normalized = name.replace(".", "_")
        lookup[f"lora_unet_{normalized}"] = module
        lookup[f"lora_unet_model_{normalized}"] = module
        if name.startswith("model."):
            lookup[f"lora_unet_{name[len('model.') :].replace('.', '_')}"] = module
    return lookup


def _get_comfy_output_axis_norm(weight):
    return weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True).reshape(weight.shape[0], *([1] * (weight.dim() - 1)))


def _reshape_dora_scale_for_comfy_output_axis(dora_scale, base_weight):
    if dora_scale.dim() == 1:
        return dora_scale.view(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))
    if dora_scale.shape[0] == base_weight.shape[0]:
        return dora_scale
    return dora_scale.reshape(-1).view(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))


def _module_state_is_lokr(module_state):
    return any(key in module_state for key in ("lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b"))


def _module_state_is_oft(module_state):
    return any(key.startswith("oft_") or key.startswith("oft_R.") for key in module_state.keys())


def _state_dict_has_native_oft_adapter(state_dict):
    return any(".oft_R.weight" in key for key in state_dict.keys())


def _resolve_lokr_scale(module_state):
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


def _convert_dora_magnitude_to_comfy_scale(module_name, module_state, base_module):
    if "lora_magnitude_vector.weight" not in module_state:
        return None

    base_weight = lora_module._get_effective_module_weight(base_module, dtype=torch.float, detach=True)
    magnitude = module_state["lora_magnitude_vector.weight"].to(device=base_weight.device, dtype=torch.float)

    if "lora_down.weight" in module_state and "lora_up.weight" in module_state:
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
    elif _module_state_is_lokr(module_state):
        scale = _resolve_lokr_scale(module_state)
        diff_weight = lokr_module._materialize_lokr_weight_from_state_dict(module_state, scale, base_weight.device)
        diff_weight = diff_weight.to(device=base_weight.device, dtype=base_weight.dtype)
        weight_norm = lokr_module._get_dokr_weight_norm(base_weight, diff_weight)
    else:
        raise ValueError(f"DoRA/DokR module {module_name} is missing supported adapter weights")

    base_norm = _get_comfy_output_axis_norm(base_weight)
    if weight_norm.is_floating_point():
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)

    magnitude = _reshape_dora_scale_for_comfy_output_axis(magnitude, base_weight)
    weight_norm = _reshape_dora_scale_for_comfy_output_axis(weight_norm, base_weight)
    dora_scale = magnitude.to(device=base_norm.device, dtype=base_norm.dtype) * (base_norm / weight_norm)
    target = module_state["lora_magnitude_vector.weight"]
    return _reshape_dora_scale_for_comfy_output_axis(dora_scale, base_weight).to(device=target.device, dtype=target.dtype)


def convert_key_to_comfy(key, preserve_alpha=False):
    """
    Convert a training format key to ComfyUI format

    Example:
        lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
        -> diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
    """
    # Standard ComfyUI LoRA ignores alpha; LoKr keeps it.
    if ".alpha" in key and not preserve_alpha:
        return None

    # Split into main part and weight part
    parts = key.split(".")
    if len(parts) < 2:
        print(f"Warning: Unexpected key format: {key}")
        return None

    main_part = parts[0]  # e.g., lora_unet_model_transformer_blocks_0_attn1_to_k
    weight_part = ".".join(parts[1:])  # e.g., lora_down.weight

    # Remove the lora_unet_ prefix and handle the wrapper's module structure.
    # Transformer keys: lora_unet_model_transformer_blocks_... (wrapper.model.transformer_blocks)
    # Connector keys:   lora_unet_embeddings_connector_... (wrapper.embeddings_connector)
    if main_part.startswith("lora_unet_model_"):
        # Standard transformer path: strip wrapper.model prefix
        main_part = main_part[len("lora_unet_model_") :]
    elif main_part.startswith("lora_unet_"):
        # Connector or other wrapper-level module
        main_part = main_part[len("lora_unet_") :]
    else:
        print(f"Warning: Key doesn't start with 'lora_unet_': {key}")
        return None

    # Map connector attribute names to ComfyUI model names
    # Training wrapper: self.embeddings_connector -> ComfyUI: video_embeddings_connector
    if main_part.startswith("embeddings_connector_"):
        main_part = "video_" + main_part
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
    converted = converted.replace("video_embeddings_connector_", "video_embeddings_connector.")
    converted = converted.replace("audio_embeddings_connector_", "audio_embeddings_connector.")
    converted = converted.replace("transformer_1d_blocks_", "transformer_1d_blocks.")

    # Step 1: Basic block structure (main transformer)
    converted = converted.replace("transformer_blocks_", "transformer_blocks.")

    # Step 2: Handle audio/video attention patterns FIRST (keep underscores in these)
    converted = converted.replace("_audio_attn1_", ".audio_attn1.")
    converted = converted.replace("_audio_attn2_", ".audio_attn2.")
    converted = converted.replace("_audio_to_video_attn_", ".audio_to_video_attn.")
    converted = converted.replace("_video_to_audio_attn_", ".video_to_audio_attn.")
    converted = converted.replace("_audio_ff_", ".audio_ff.")

    # Step 3: Now handle regular (non-audio) attention patterns
    converted = converted.replace("_attn1_", ".attn1.")
    converted = converted.replace("_attn2_", ".attn2.")

    # Step 4: Handle projection layers (use regex to avoid matching inside audio_to_video, video_to_audio)
    # Only match _to_k/_to_q/_to_v at word boundaries (end of string or followed by .)
    converted = re.sub(r"_to_k($|\.)", r".to_k\1", converted)
    converted = re.sub(r"_to_q($|\.)", r".to_q\1", converted)
    converted = re.sub(r"_to_v($|\.)", r".to_v\1", converted)
    converted = re.sub(r"_to_gate_logits($|\.)", r".to_gate_logits\1", converted)
    converted = re.sub(r"_to_out\.", r".to_out.", converted)
    converted = re.sub(r"_to_out_", r".to_out.", converted)

    # Step 5: Handle feedforward layers
    converted = converted.replace("_ff_net_", ".ff.net.")
    converted = converted.replace("_ff_", ".ff.")
    converted = converted.replace("_net_", ".net.")
    converted = converted.replace("_proj", ".proj")

    # Step 6: Fix remaining numeric suffixes
    converted = re.sub(r"to_out_(\d+)", r"to_out.\1", converted)
    converted = re.sub(r"net_(\d+)", r"net.\1", converted)

    # Build the final key
    comfy_key = f"diffusion_model.{converted}"

    # Convert weight naming: lora_down -> lora_A, lora_up -> lora_B
    if "lora_down" in weight_part:
        weight_part = weight_part.replace("lora_down", "lora_A")
    elif "lora_up" in weight_part:
        weight_part = weight_part.replace("lora_up", "lora_B")
    elif weight_part == "lora_magnitude_vector.weight":
        weight_part = "dora_scale"

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
        weight_part = "lora_magnitude_vector.weight"
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
    elif key.endswith(".lokr_w1"):
        weight_part = "lokr_w1"
        path = key[: -len(".lokr_w1")]
    elif key.endswith(".lokr_w2"):
        weight_part = "lokr_w2"
        path = key[: -len(".lokr_w2")]
    elif key.endswith(".lokr_w2_a"):
        weight_part = "lokr_w2_a"
        path = key[: -len(".lokr_w2_a")]
    elif key.endswith(".lokr_w2_b"):
        weight_part = "lokr_w2_b"
        path = key[: -len(".lokr_w2_b")]
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
    """Return True if the state dict looks like an LTX-2 ComfyUI LoRA."""
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
    return any(key.startswith("diffusion_model.") and key.endswith(recognized_suffixes) for key in weights_sd.keys())


def _is_ltx2_gate_logits_module(module_name: str) -> bool:
    return module_name.endswith("_to_gate_logits")


def _is_ltx2_feed_forward_module(module_name: str) -> bool:
    return "_ff_" in module_name


def _should_skip_comfy_dora_scale(module_name: str, *, skip_gate_logits_dora: bool, dora_ff_only: bool) -> bool:
    if dora_ff_only and not _is_ltx2_feed_forward_module(module_name):
        return True
    if skip_gate_logits_dora and _is_ltx2_gate_logits_module(module_name):
        return True
    return False


def convert_lora_to_comfy_state_dict(
    trained_state_dict,
    verbose=False,
    network=None,
    base_model=None,
    skip_gate_logits_dora=False,
    dora_ff_only=False,
):
    """Convert a training-format LTX-2 LoRA state dict to ComfyUI format."""
    # Collect alpha and rank per LoRA module to fold scale into weights
    lora_alpha = {}
    lora_rank = {}
    for key, tensor in trained_state_dict.items():
        if key.endswith(".lora_down.weight"):
            lora_name = key.rsplit(".", 2)[0]
            if lora_name not in lora_rank:
                lora_rank[lora_name] = tensor.shape[0]
        elif key.endswith(".alpha"):
            lora_name = key.rsplit(".", 1)[0]
            lora_alpha[lora_name] = tensor

    grouped_modules = _group_lora_keys_by_module(trained_state_dict)
    lokr_modules = {module_name for module_name, module_state in grouped_modules.items() if _module_state_is_lokr(module_state)}
    dora_modules = [
        module_name for module_name, module_state in grouped_modules.items() if "lora_magnitude_vector.weight" in module_state
    ]
    dora_scales = {}
    if dora_modules:
        base_modules = _build_lora_base_module_lookup(network)
        base_modules.update(_build_lora_module_lookup(base_model))
        modules_needing_base = [
            module_name
            for module_name in dora_modules
            if not _should_skip_comfy_dora_scale(
                module_name,
                skip_gate_logits_dora=skip_gate_logits_dora,
                dora_ff_only=dora_ff_only,
            )
        ]
        missing = [module_name for module_name in modules_needing_base if module_name not in base_modules]
        if missing:
            missing_list = ", ".join(missing[:3])
            if len(missing) > 3:
                missing_list += ", ..."
            raise ValueError(
                "Converting DoRA LoRA to ComfyUI requires the live LoRA network so "
                f"native magnitude vectors can be translated to dora_scale. Missing base modules: {missing_list}"
            )
        for module_name in dora_modules:
            module_state = grouped_modules[module_name]
            is_stock_comfy_dokr = _module_state_is_lokr(module_state) and not _module_state_is_oft(module_state)
            if dora_ff_only and not _is_ltx2_feed_forward_module(module_name):
                trained_state_dict = dict(trained_state_dict)
                trained_state_dict.pop(f"{module_name}.lora_magnitude_vector.weight", None)
                if is_stock_comfy_dokr:
                    trained_state_dict.pop(f"{module_name}.initial_norm", None)
                if verbose:
                    print(f"Skipped ComfyUI DoRA scale for non-FF LTX-2 module: {module_name}")
                continue
            if skip_gate_logits_dora and _is_ltx2_gate_logits_module(module_name):
                trained_state_dict = dict(trained_state_dict)
                trained_state_dict.pop(f"{module_name}.lora_magnitude_vector.weight", None)
                if is_stock_comfy_dokr:
                    trained_state_dict.pop(f"{module_name}.initial_norm", None)
                if verbose:
                    print(f"Skipped ComfyUI DoRA scale for LTX-2 gate logits: {module_name}")
                continue
            magnitude_key = f"{module_name}.lora_magnitude_vector.weight"
            dora_scales[magnitude_key] = _convert_dora_magnitude_to_comfy_scale(
                module_name,
                module_state,
                base_modules[module_name],
            )

    # Convert keys
    comfy_state_dict = {}
    skipped_alpha = 0
    folded_alpha = 0
    converted_dora = 0
    converted = 0
    failed = 0

    for key, tensor in trained_state_dict.items():
        lora_name = key.rsplit(".", 1)[0] if key.endswith(".alpha") else key.split(".", 1)[0]
        preserve_alpha = key.endswith(".alpha") and lora_name in lokr_modules
        new_key = convert_key_to_comfy(key, preserve_alpha=preserve_alpha)

        if new_key is None:
            if ".alpha" in key:
                skipped_alpha += 1
                if verbose:
                    print(f"Skipping alpha key: {key}")
            else:
                failed += 1
                print(f"Failed to convert key: {key}")
        else:
            # Fold alpha scale into lora_B (up) weights, since Comfy ignores alpha keys.
            if key.endswith(".lora_up.weight"):
                lora_name = key.rsplit(".", 2)[0]
                alpha = lora_alpha.get(lora_name, None)
                rank = lora_rank.get(lora_name, None)
                if alpha is not None and rank is not None and rank != 0:
                    scale = float(alpha.item()) / float(rank)
                    tensor = tensor * scale
                    folded_alpha += 1
                    if verbose:
                        print(f"Folded alpha for {lora_name}: alpha={float(alpha.item())} rank={rank} scale={scale}")
            elif key in dora_scales:
                tensor = dora_scales[key]
                converted_dora += 1
                if verbose:
                    print(f"Translated DoRA magnitude to ComfyUI dora_scale for {key.rsplit('.', 2)[0]}")
            comfy_state_dict[new_key] = tensor
            converted += 1
            if verbose:
                print(f"Converted: {key} -> {new_key}")

    if verbose:
        print("\nConversion summary:")
        print(f"  Converted: {converted} keys")
        print(f"  Skipped alpha keys: {skipped_alpha}")
        print(f"  Folded alpha into lora_B: {folded_alpha}")
        print(f"  Translated DoRA dora_scale: {converted_dora}")
        print(f"  Failed: {failed} keys")
        print(f"  Output LoRA has {len(comfy_state_dict)} keys")

    return comfy_state_dict


def convert_lora_from_comfy_state_dict(comfy_state_dict):
    """
    Convert a ComfyUI-format LTX-2 LoRA state dict back to training format.

    Since ComfyUI checkpoints do not store alpha separately, this recreates
    native ``.alpha`` buffers with ``alpha=rank``. This preserves the effective
    LoRA delta when the checkpoint is warm-started for further training.
    """
    if any(key.startswith("diffusion_model.") and key.endswith(".dora_scale") for key in comfy_state_dict.keys()):
        raise ValueError(
            "ComfyUI DoRA dora_scale cannot be converted back to native Musubi magnitude vectors without base weights. "
            "Use the original Musubi DoRA checkpoint for Musubi inference or resume."
        )

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
    network=None,
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
        network: Live LoRA network, required for DoRA magnitude conversion

    Returns:
        Path to the output file
    """
    print(f"Loading LoRA from: {input_path}")
    loaded_base_model = False
    trained_state_dict = None
    comfy_state_dict = None

    try:
        trained_state_dict = safetensors.torch.load_file(input_path)
        print(f"Input LoRA has {len(trained_state_dict)} keys")

        needs_base = (
            base_model is None
            and network is None
            and any(".lora_magnitude_vector.weight" in key for key in trained_state_dict.keys())
        )
        if needs_base and base_model_path is not None:
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
            network=network,
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

        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.comfy{input_file.suffix}"

        metadata = None
        try:
            with safetensors.safe_open(input_path, framework="pt") as f:
                metadata = f.metadata()
            if metadata:
                print(f"Preserving {len(metadata)} metadata entries")
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")

        print(f"\nSaving ComfyUI-compatible LoRA to: {output_path}")
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
    parser = argparse.ArgumentParser(description="Convert LTX-2 LoRA from training format to ComfyUI format")
    parser.add_argument("input", type=str, help="Path to the input LoRA file (training format)")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to save the converted LoRA (default: <input>.comfy.safetensors)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed conversion information")
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
    parser.add_argument("--audio_video", action="store_true", help="Load the audio-video LTX-2 transformer variant.")
    parser.add_argument("--audio_only_model", action="store_true", help="Load the audio-only LTX-2 transformer variant.")
    parser.add_argument("--base_dtype", type=str, default="float32", help="Dtype used when loading the base transformer.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used when loading the base transformer.")
    parser.add_argument("--fp8_base", action="store_true", help="Load base transformer weights in non-scaled FP8 mode.")
    parser.add_argument("--fp8_scaled", action="store_true", help="Load base transformer weights using scaled FP8.")
    parser.add_argument("--fp8_w8a8", action="store_true", help="Apply W8A8 patch after scaled FP8 loading.")
    parser.add_argument("--w8a8_mode", type=str, default="int8", choices=["int8", "fp8"], help="W8A8 format.")
    parser.add_argument("--fp8_keep_blocks", type=str, default=None, help="FP8 high-precision transformer blocks.")
    parser.add_argument("--nf4_base", action="store_true", help="Load base transformer weights using NF4.")
    parser.add_argument("--nf4_block_size", type=int, default=32, help="NF4 block size.")
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
