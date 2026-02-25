"""FP8 utilities for loading quantized Gemma safetensors (e.g. ComfyUI fp8_e4m3fn exports)."""

import json
import logging
import re
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FP8Linear(nn.Module):
    """Drop-in Linear replacement that keeps weights in fp8 and dequantizes on forward."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        *,
        weight_offload: bool = False,
    ):
        super().__init__()
        self.weight_offload = bool(weight_offload)
        if self.weight_offload:
            self.weight = nn.Parameter(weight.detach().to("cpu"), requires_grad=False)
            self.bias = nn.Parameter(bias.detach().to("cpu", dtype=compute_dtype), requires_grad=False) if bias is not None else None
        else:
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
        self.compute_dtype = compute_dtype
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype if x.dtype.is_floating_point else self.compute_dtype

        if self.weight_offload:
            w = self.weight.to(device=x.device, dtype=target_dtype, non_blocking=True)
            b = self.bias.to(device=x.device, dtype=target_dtype, non_blocking=True) if self.bias is not None else None
        else:
            w = self.weight
            if w.device != x.device or w.dtype != target_dtype:
                w = w.to(device=x.device, dtype=target_dtype)

            b = self.bias
            if b is not None and (b.device != x.device or b.dtype != target_dtype):
                b = b.to(device=x.device, dtype=target_dtype)

        return F.linear(x, w, b)


def replace_linear_with_fp8(model: nn.Module, compute_dtype: torch.dtype, *, weight_offload: bool = False) -> int:
    """Walk module tree and replace nn.Linear modules whose weights are fp8 with FP8Linear.

    Returns the number of modules replaced.
    """
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            replacement = FP8Linear(
                child.weight.data,
                child.bias.data if child.bias is not None else None,
                compute_dtype,
                weight_offload=weight_offload,
            )
            setattr(model, name, replacement)
            count += 1
        else:
            count += replace_linear_with_fp8(child, compute_dtype, weight_offload=weight_offload)
    return count


# ---------------------------------------------------------------------------
# Known Gemma 3 attention head configurations keyed by hidden_size.
# (num_attention_heads, num_key_value_heads, head_dim)
#
# These can't be uniquely determined from tensor shapes alone because
# multiple factorizations of q_proj output dimension are possible
# (e.g. 4096 = 16*256 = 32*128).
# ---------------------------------------------------------------------------
_GEMMA3_HEAD_CONFIG: dict[int, tuple[int, int, int]] = {
    1536: (8, 4, 256),    # Gemma 3 1B
    2560: (8, 4, 256),    # Gemma 3 4B
    3840: (16, 8, 256),   # Gemma 3 12B
    5376: (32, 16, 128),  # Gemma 3 27B
}


def infer_gemma3_config_from_safetensors(path: str) -> dict:
    """Read safetensors header and infer Gemma 3 architecture config from tensor shapes.

    Reads only the file header (fast, no weight data loaded) and constructs
    a config dict compatible with ``Gemma3ForConditionalGeneration.config_class``.
    Works for any Gemma 3 model size (1B, 4B, 12B, 27B).
    """
    # Read safetensors header — JSON metadata with tensor names, shapes, dtypes
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    def find_shape(pattern: str) -> list[int] | None:
        for key, info in tensors.items():
            if pattern in key:
                return info["shape"]
        return None

    # --- hidden_size and vocab_size from embedding ---
    embed_shape = find_shape("embed_tokens.weight")
    if embed_shape is None:
        raise ValueError(
            f"Cannot find embed_tokens.weight in {path}. "
            "This doesn't look like a Gemma safetensors file."
        )
    vocab_size, hidden_size = embed_shape

    # --- intermediate_size from gate_proj ---
    gate_shape = find_shape("mlp.gate_proj.weight")
    intermediate_size = gate_shape[0] if gate_shape else hidden_size * 4

    # --- num_hidden_layers from counting layer indices ---
    layer_indices: set[int] = set()
    for key in tensors:
        m = re.search(r"layers\.(\d+)\.", key)
        if m:
            layer_indices.add(int(m.group(1)))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 48

    # --- attention head configuration ---
    if hidden_size in _GEMMA3_HEAD_CONFIG:
        num_attention_heads, num_key_value_heads, head_dim = _GEMMA3_HEAD_CONFIG[hidden_size]
    else:
        # Unknown model size — try to infer from q_proj/k_proj shapes
        q_shape = find_shape(".self_attn.q_proj.weight")
        k_shape = find_shape(".self_attn.k_proj.weight")
        if q_shape and k_shape:
            q_dim, k_dim = q_shape[0], k_shape[0]
            # Try head_dim candidates (256 is most common, 128 for 27B)
            for candidate_hd in (256, 128, 64):
                if q_dim % candidate_hd == 0 and k_dim % candidate_hd == 0:
                    head_dim = candidate_hd
                    num_attention_heads = q_dim // head_dim
                    num_key_value_heads = k_dim // head_dim
                    break
            else:
                raise ValueError(
                    f"Cannot determine head_dim from q_proj={q_dim}, k_proj={k_dim}. "
                    "Use --gemma_root with a config.json instead."
                )
        else:
            raise ValueError(
                f"Unknown Gemma 3 model (hidden_size={hidden_size}) and no attention "
                "projections found. Use --gemma_root with a config.json instead."
            )

    logger.info(
        "Inferred Gemma 3 config from safetensors: hidden_size=%d, "
        "intermediate_size=%d, layers=%d, heads=%d, kv_heads=%d, "
        "head_dim=%d, vocab_size=%d",
        hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, vocab_size,
    )

    return {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "hidden_activation": "gelu_pytorch_tanh",
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "attention_bias": False,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
        },
        "vision_config": {
            # SigLIP vision tower — same across all Gemma 3 multimodal sizes
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "image_size": 896,
            "patch_size": 14,
        },
        "mm_tokens_per_image": 256,
        "image_token_index": 262145,
    }
