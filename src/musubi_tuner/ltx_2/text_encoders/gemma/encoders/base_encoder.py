import functools
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

import torch
from einops import rearrange
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor
from musubi_tuner.ltx_2.loader.module_ops import ModuleOps
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from musubi_tuner.ltx_2.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer


class GemmaTextEncoderModelBase(torch.nn.Module):
    """
    Gemma Text Encoder Model.
    This base class combines the tokenizer, Gemma model and feature extractor to provide a preprocessing
    for implementation classes for multimodal pipelines. It processes input text through tokenization,
    obtains hidden states from the base language model, applies a linear feature extractor.
    Args:
        tokenizer (LTXVGemmaTokenizer): The tokenizer used for text preprocessing.
        model (Gemma3ForConditionalGeneration): The base Gemma LLM.
        feature_extractor_linear (torch.nn.Module): Text feature extractor module.
        dtype (torch.dtype, optional): The data type for model parameters (default: torch.bfloat16).
    """

    def __init__(
        self,
        feature_extractor_linear: torch.nn.Module,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        img_processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._gemma_root = None
        self.tokenizer = tokenizer
        self.model = model
        self.processor = img_processor
        self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)

    def _text_model_device(self) -> torch.device:
        model = getattr(self, "model", None)
        if model is None:
            return torch.device("cpu")

        try:
            embeddings = model.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                return embeddings.weight.device
        except Exception:
            pass

        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _run_feature_extractor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        return self.feature_extractor_linear(hidden_states, attention_mask, padding_side=padding_side)

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(torch.int64)
        return (attention_mask - 1).to(dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(
            dtype
        ).max

    def _preprocess_text(
        self, text: str, padding_side: str = "left"
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor | None], torch.Tensor]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.
        Args:
            text (str): Input string to encode.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        text_device = self._text_model_device()
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=text_device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=text_device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = self._run_feature_extractor(
            hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    def _init_image_processor(self) -> None:
        img_processor = AutoImageProcessor.from_pretrained(self._gemma_root, local_files_only=True, use_fast=False)
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded, cannot load image processor")
        self.processor = Gemma3Processor(image_processor=img_processor, tokenizer=self.tokenizer.tokenizer)

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> str:
        if self.processor is None:
            self._init_image_processor()
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_device = self._text_model_device()
        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(text_device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        rng_devices = [text_device] if text_device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for T2V generation."""

        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    def forward(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is not implemented for the base class")

    def to(self, *args, **kwargs):
        model = getattr(self, "model", None)
        protect_model = False
        if model is not None:
            protect_model = (
                bool(getattr(model, "is_loaded_in_8bit", False))
                or bool(getattr(model, "is_loaded_in_4bit", False))
                or bool(getattr(self, "_has_fp8_model", False))
            )

        if not protect_model:
            return super().to(*args, **kwargs)

        stored_model = self._modules.pop("model", None)
        try:
            return super().to(*args, **kwargs)
        finally:
            if stored_model is not None:
                self._modules["model"] = stored_model
                self.model = stored_model


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(t, device=device)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + eps)

    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(b, t, -1)  # [B, T, D * L]

    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _find_matching_dir(root_path: str, pattern: str) -> str:
    """
    Recursively search for files matching a glob pattern and return the parent directory of the first match.
    """

    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


def _infer_safetensors_dtype(path: str) -> torch.dtype | None:
    with MemoryEfficientSafeOpen(path) as handle:
        for key in handle.keys():
            meta = handle.header.get(key)
            if not isinstance(meta, dict) or "dtype" not in meta:
                continue
            dt = handle._get_torch_dtype(meta["dtype"])  # noqa: SLF001
            if isinstance(dt, torch.dtype) and dt.is_floating_point:
                return dt
    return None


def _has_fp8_weights(path: str) -> bool:
    """Check if a safetensors file contains any fp8 tensors."""
    with MemoryEfficientSafeOpen(path) as handle:
        for key in handle.keys():
            meta = handle.header.get(key)
            if not isinstance(meta, dict) or "dtype" not in meta:
                continue
            if meta["dtype"] in ("F8_E5M2", "F8_E4M3"):
                return True
    return False


def _extract_spiece_model_bytes(safetensors_path: str) -> bytes:
    """Extract spiece_model tokenizer bytes from a safetensors file."""
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        if "spiece_model" not in f.keys():
            raise ValueError(
                f"No 'spiece_model' key found in {safetensors_path}. "
                "Cannot extract tokenizer. Provide --gemma_root for the tokenizer."
            )
        tensor = f.get_tensor("spiece_model")
        return tensor.numpy().tobytes()


def _resolve_module_and_attr(root_module: torch.nn.Module, dotted_name: str) -> tuple[torch.nn.Module, str]:
    parts = dotted_name.split(".")
    submodule = root_module
    for part in parts[:-1]:
        submodule = getattr(submodule, part)
    return submodule, parts[-1]


def _materialize_meta_parameter(
    root_module: torch.nn.Module,
    dotted_name: str,
    reference: torch.nn.Parameter,
    *,
    device: torch.device,
) -> None:
    submodule, attr_name = _resolve_module_and_attr(root_module, dotted_name)
    replacement = torch.nn.Parameter(
        torch.zeros(reference.shape, dtype=reference.dtype, device=device),
        requires_grad=reference.requires_grad,
    )
    setattr(submodule, attr_name, replacement)


def _materialize_meta_buffer(
    root_module: torch.nn.Module,
    dotted_name: str,
    reference: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    submodule, attr_name = _resolve_module_and_attr(root_module, dotted_name)

    # Gemma3 keeps some runtime-only buffers as persistent=False; these are validly absent from checkpoints.
    if attr_name == "inv_freq" and hasattr(submodule, "rope_init_fn") and hasattr(submodule, "config"):
        try:
            inv_freq, attention_scaling = submodule.rope_init_fn(submodule.config, device)
            replacement = inv_freq.to(device=device)
            setattr(submodule, attr_name, replacement)
            if hasattr(submodule, "original_inv_freq"):
                submodule.original_inv_freq = replacement
            if hasattr(submodule, "attention_scaling"):
                submodule.attention_scaling = attention_scaling
            return
        except Exception as e:
            logger.warning("Failed to rebuild rotary buffer %s: %s", dotted_name, e)

    if attr_name == "embed_scale" and hasattr(submodule, "embedding_dim"):
        replacement = torch.tensor(float(submodule.embedding_dim**0.5), dtype=reference.dtype, device=device)
        setattr(submodule, attr_name, replacement)
        return

    replacement = torch.zeros(reference.shape, dtype=reference.dtype, device=device)
    setattr(submodule, attr_name, replacement)


_QUANTIZED_LINEAR_CLASS_NAMES = frozenset({"Linear4bit", "Linear8bitLt", "FP8Linear"})


def _override_target_is_quantized(submodule: torch.nn.Module, attr_name: str) -> bool:
    if type(submodule).__name__ in _QUANTIZED_LINEAR_CLASS_NAMES:
        return True
    current = getattr(submodule, attr_name, None)
    if isinstance(current, torch.Tensor) and current.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    return False


def apply_text_encoder_checkpoint_overrides(
    text_encoder: torch.nn.Module,
    checkpoint_path: str | None,
    *,
    prefix: str = "text_encoder.",
) -> int:
    """Overlay text-encoder weights stored inside a training checkpoint.

    Full-FT checkpoints save Gemma weights under the ``text_encoder.`` prefix.
    The runtime Gemma loader still rebuilds the module from ``gemma_root`` /
    ``gemma_safetensors`` plus the checkpoint connectors, so we need to copy any
    prefixed overrides on top of the rebuilt module to restore the finetuned
    Gemma state.

    Overrides whose target module is quantized (bnb 4/8-bit) or fp8-wrapped
    cannot be assigned as plain tensors, so they are skipped with an aggregated
    warning; unquantized overrides (embeddings, layernorms, etc.) are still
    applied.
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0

    override_keys: list[str] = []
    with MemoryEfficientSafeOpen(checkpoint_path) as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                override_keys.append(key)

        if not override_keys:
            return 0

        loaded = 0
        skipped_quantized: list[str] = []
        for key in override_keys:
            module_key = key[len(prefix) :]
            try:
                submodule, attr_name = _resolve_module_and_attr(text_encoder, module_key)
            except AttributeError:
                logger.warning("Ignoring unknown text-encoder override key from checkpoint: %s", key)
                continue

            if _override_target_is_quantized(submodule, attr_name):
                skipped_quantized.append(module_key)
                continue

            current_value = getattr(submodule, attr_name)
            tensor = handle.get_tensor(key)

            if isinstance(current_value, torch.nn.Parameter):
                if tuple(current_value.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Text-encoder override shape mismatch for {module_key}: "
                        f"checkpoint={tuple(tensor.shape)} runtime={tuple(current_value.shape)}"
                    )
                replacement = torch.nn.Parameter(
                    tensor.to(device=current_value.device, dtype=current_value.dtype),
                    requires_grad=current_value.requires_grad,
                )
                setattr(submodule, attr_name, replacement)
                loaded += 1
                continue

            if isinstance(current_value, torch.Tensor):
                if tuple(current_value.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Text-encoder override shape mismatch for {module_key}: "
                        f"checkpoint={tuple(tensor.shape)} runtime={tuple(current_value.shape)}"
                    )
                setattr(submodule, attr_name, tensor.to(device=current_value.device, dtype=current_value.dtype))
                loaded += 1
                continue

            logger.warning("Ignoring non-tensor text-encoder override target %s (%s)", module_key, type(current_value).__name__)

    if skipped_quantized:
        logger.warning(
            "Checkpoint contains %d finetuned text_encoder override(s) whose target modules are "
            "quantized/fp8 at runtime; they were NOT applied and those layers will use the base "
            "Gemma weights. Load Gemma in full precision (no --gemma_load_in_4bit/8bit, non-fp8 "
            "weights) to restore them. Affected keys (first 10): %s",
            len(skipped_quantized),
            skipped_quantized[:10],
        )

    if loaded > 0:
        logger.info("Applied %d finetuned text-encoder tensors from checkpoint: %s", loaded, checkpoint_path)
    return loaded


def module_ops_from_gemma_root(
    gemma_root: str | None,
    *,
    gemma_weights_path: str | None = None,
    gemma_safetensors: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: torch.dtype | None = None,
    bnb_device_map=None,
    fp8_weight_offload: bool | None = None,
    device: torch.device | None = None,
) -> tuple[ModuleOps, ...]:
    # -- gemma_safetensors preprocessing --
    keep_fp8 = False
    if gemma_safetensors:
        if load_in_8bit or load_in_4bit:
            raise ValueError("--gemma_safetensors cannot be combined with --gemma_load_in_4bit/8bit")
        sf_path = Path(gemma_safetensors)
        if not sf_path.exists():
            raise FileNotFoundError(f"Gemma safetensors not found: {gemma_safetensors}")
        gemma_weights_path = gemma_safetensors
        gemma_root = None
        keep_fp8 = _has_fp8_weights(gemma_safetensors)
        if keep_fp8:
            logger.info("Detected fp8 weights in %s — will keep fp8 in VRAM", gemma_safetensors)

    if os.getenv("LTX2_REQUIRE_GEMMA_ROOT", "0") == "1" and gemma_root is None and not gemma_safetensors:
        raise ValueError("gemma_root is required for this configuration")
    gemma_weights_dtype = None
    if gemma_weights_path:
        weight_path = Path(gemma_weights_path)
        if weight_path.is_dir():
            gemma_path = _find_matching_dir(str(weight_path), "model*.safetensors")
            gemma_weights_path = None
        else:
            if not weight_path.exists():
                raise FileNotFoundError(f"Gemma weights not found: {gemma_weights_path}")
            gemma_weights_dtype = _infer_safetensors_dtype(str(weight_path))
            gemma_path = str(weight_path.parent)

        if gemma_root is None and not gemma_safetensors:
            gemma_root = gemma_path
    elif gemma_root is not None:
        gemma_path = _find_matching_dir(gemma_root, "model*.safetensors")
    else:
        raise ValueError("Either gemma_root, gemma_weights_path, or gemma_safetensors must be provided")

    # Resolve tokenizer: from gemma_root directory or extracted from safetensors
    if gemma_root is not None:
        tokenizer_path: str | bytes = _find_matching_dir(gemma_root, "tokenizer.model")
    elif gemma_safetensors:
        logger.info("Extracting tokenizer from safetensors file...")
        tokenizer_path = _extract_spiece_model_bytes(gemma_safetensors)
    else:
        raise ValueError("Cannot resolve tokenizer: provide --gemma_root or --gemma_safetensors")

    def load_gemma(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be enabled")

        if load_in_8bit or load_in_4bit:
            if not torch.cuda.is_available():
                raise ValueError("8-bit/4-bit Gemma loading requires CUDA")
            if gemma_weights_path is not None:
                raise ValueError("gemma_weights_path is not supported with 8-bit/4-bit loading")

            from transformers import BitsAndBytesConfig

            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                compute_dtype = bnb_4bit_compute_dtype if bnb_4bit_compute_dtype is not None else torch_dtype
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

            module.model = Gemma3ForConditionalGeneration.from_pretrained(
                gemma_path,
                local_files_only=True,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map=bnb_device_map if bnb_device_map is not None else {"": "cuda"},
            )
        else:
            if gemma_weights_path is not None:
                from safetensors import safe_open

                load_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                if keep_fp8 and load_device.type != "cuda":
                    raise ValueError("Float8 Gemma weights require CUDA; provide a GPU or use non-fp8 weights.")
                elif gemma_weights_dtype is not None and gemma_weights_dtype.itemsize == 1 and load_device.type != "cuda":
                    raise ValueError("Float8 Gemma weights require CUDA; provide a GPU or use non-fp8 weights.")

                if gemma_root is not None:
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(gemma_root, local_files_only=True)
                else:
                    from musubi_tuner.ltx_2.text_encoders.gemma.fp8_ops import infer_gemma3_config_from_safetensors

                    logger.info("No gemma_root — inferring config from safetensors header...")
                    inferred = infer_gemma3_config_from_safetensors(gemma_safetensors)
                    config_class = Gemma3ForConditionalGeneration.config_class
                    config = config_class(**inferred)

                # Initialize on meta device to avoid immediate allocation
                with torch.device("meta"):
                    module.model = Gemma3ForConditionalGeneration(config).to(dtype=torch_dtype)

                logger.info(f"Loading custom Gemma weights from {gemma_weights_path}...")

                # Memory-efficient loading: stream tensors directly to model
                with safe_open(gemma_weights_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    total_keys = len(keys)
                    logger.info(f"Found {total_keys} tensors in safetensors file")

                    # Print first 5 keys from safetensors for debugging
                    logger.info(f"First 5 safetensors keys: {keys[:5]}")

                    # Get model's expected keys for comparison
                    model_keys = set(name for name, _ in module.model.named_parameters())
                    logger.info(f"Model expects {len(model_keys)} parameters")
                    logger.info(f"First 5 model keys: {list(model_keys)[:5]}")

                    unmatched_keys = []

                    for i, key in enumerate(keys):
                        if i % 100 == 0:
                            logger.info(f"Loading Gemma weights: {i}/{total_keys}...")

                        # Skip ComfyUI quantization metadata keys (not actual weights)
                        if key.endswith(".comfy_quant") or key.endswith(".weight_scale") or key.endswith(".weight_scale_2"):
                            continue

                        new_key = key

                        # Key mapping for ComfyUI/flattened checkpoints
                        # ComfyUI uses model.* but HuggingFace Gemma3ForConditionalGeneration uses model.language_model.*
                        if key.startswith("model.embed_tokens."):
                            new_key = key.replace("model.embed_tokens.", "model.language_model.embed_tokens.", 1)
                        elif key.startswith("model.layers."):
                            new_key = key.replace("model.layers.", "model.language_model.layers.", 1)
                        elif key.startswith("model.norm."):
                            new_key = key.replace("model.norm.", "model.language_model.norm.", 1)
                        elif key.startswith("vision_model."):
                            new_key = f"model.vision_tower.{key}"
                        elif key.startswith("multi_modal_projector."):
                            new_key = f"model.{key}"
                        elif key.startswith("language_model."):
                            # Some checkpoints use language_model.* instead of model.language_model.*
                            new_key = f"model.{key}"

                        try:
                            # Iterate to find the submodule and parameter
                            sub_mod, param_name = _resolve_module_and_attr(module.model, new_key)
                            param = getattr(sub_mod, param_name)

                            # Skip if already loaded (unlikely in this loop but good safety)
                            if param.device.type != "meta":
                                pass

                            tensor = f.get_tensor(key)

                            with torch.no_grad():
                                # Dequantize NVFP4 packed weights (uint8 with two-level scales)
                                if tensor.dtype == torch.uint8 and key.endswith(".weight"):
                                    prefix = key[:-len(".weight")]
                                    wscale_k = prefix + ".weight_scale"
                                    wscale2_k = prefix + ".weight_scale_2"
                                    if wscale_k in keys and wscale2_k in keys:
                                        from musubi_tuner.modules.nvfp4_utils import dequantize_nvfp4_weight
                                        block_scale = f.get_tensor(wscale_k)
                                        tensor_scale = f.get_tensor(wscale2_k)
                                        tensor = dequantize_nvfp4_weight(tensor, block_scale, tensor_scale, dtype=torch_dtype)
                                # Dequantize ComfyUI FP8 weights with per-tensor scale
                                elif tensor.dtype.itemsize == 1 and tensor.dtype != torch.uint8 and key.endswith(".weight"):
                                    wscale_k = key.replace(".weight", ".weight_scale")
                                    if wscale_k in keys:
                                        scale = f.get_tensor(wscale_k)
                                        tensor = tensor.to(torch_dtype) * scale.to(torch_dtype)

                                if param.shape != tensor.shape:
                                    # Handle specialized shape mismatches if necessary, or error
                                    logger.warning(f"Shape mismatch for {new_key}: model {param.shape} vs ckpt {tensor.shape}")
                                    continue

                                if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not keep_fp8:
                                    # Cast fp8→compute dtype when not keeping fp8 in VRAM
                                    tensor = tensor.to(torch_dtype)

                                # Materialize param on target device
                                new_param = torch.nn.Parameter(tensor.to(device=load_device), requires_grad=param.requires_grad)
                                # Replace the meta parameter with the materialized one
                                setattr(sub_mod, param_name, new_param)

                        except AttributeError:
                            # Missing in model (unexpected key) - track for debugging
                            if i < 20:  # Only log first 20 unmatched
                                unmatched_keys.append((key, new_key))
                            pass
                        except Exception as e:
                            logger.warning(f"Error loading {new_key}: {e}")
                # Log unmatched keys for debugging
                if unmatched_keys:
                    logger.info(f"First {len(unmatched_keys)} unmatched safetensors keys (original -> attempted):")
                    for orig, attempted in unmatched_keys:
                        logger.info(f"  {orig} -> {attempted}")

                # Some text-encoder exports omit lm_head; tie it to input embeddings before meta checks.
                if hasattr(module.model, "tie_weights"):
                    try:
                        module.model.tie_weights()
                    except Exception as e:
                        logger.warning("Failed to tie Gemma lm_head weights: %s", e)

                meta_params = [(name, p) for name, p in module.model.named_parameters() if p.device.type == "meta"]
                meta_buffers = [(name, b) for name, b in module.model.named_buffers() if b.device.type == "meta"]
                total_params = sum(1 for _ in module.model.parameters())
                logger.info(
                    "Loaded %d/%d parameters from safetensors. %d parameters and %d buffers remain on meta.",
                    total_params - len(meta_params),
                    total_params,
                    len(meta_params),
                    len(meta_buffers),
                )

                required_prefixes = ("model.language_model.", "lm_head.")
                missing_required_params = [name for name, _ in meta_params if name.startswith(required_prefixes)]
                missing_required_buffers = [name for name, _ in meta_buffers if name.startswith(required_prefixes)]
                derivable_required_buffer_suffixes = (".embed_scale", ".inv_freq")
                non_derivable_required_buffers = [
                    name for name in missing_required_buffers if not name.endswith(derivable_required_buffer_suffixes)
                ]

                if missing_required_params or non_derivable_required_buffers:
                    raise ValueError(
                        "Gemma safetensors is missing required language-model tensors. "
                        f"missing_params={missing_required_params[:20]} "
                        f"missing_buffers={non_derivable_required_buffers[:20]}"
                    )

                if meta_params or meta_buffers:
                    # Optional multimodal branches can be absent in text-focused checkpoints.
                    # Materialize them as zeros on CPU to avoid full-GPU allocation.
                    optional_device = torch.device("cpu") if load_device.type == "cuda" else load_device
                    if missing_required_buffers:
                        logger.info(
                            "Rebuilding %d language-model runtime buffers from config: %s",
                            len(missing_required_buffers),
                            missing_required_buffers[:10],
                        )
                    for name, param in meta_params:
                        target_device = load_device if name.startswith(required_prefixes) else optional_device
                        _materialize_meta_parameter(module.model, name, param, device=target_device)
                    for name, buf in meta_buffers:
                        target_device = load_device if name.startswith(required_prefixes) else optional_device
                        _materialize_meta_buffer(module.model, name, buf, device=target_device)
                    logger.info(
                        "Materialized %d missing parameters and %d buffers (optional_device=%s, required_device=%s)",
                        len(meta_params),
                        len(meta_buffers),
                        optional_device,
                        load_device,
                    )

                if keep_fp8:
                    from musubi_tuner.ltx_2.text_encoders.gemma.fp8_ops import replace_linear_with_fp8

                    if fp8_weight_offload is None:
                        offload_fp8_weights = os.getenv("LTX2_GEMMA_SAFETENSORS_WEIGHT_OFFLOAD", "1").lower() in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        )
                    else:
                        offload_fp8_weights = bool(fp8_weight_offload)
                    n_replaced = replace_linear_with_fp8(
                        module.model,
                        torch_dtype,
                        weight_offload=offload_fp8_weights,
                    )
                    logger.info(
                        "Replaced %d Linear modules with FP8Linear (weight_offload=%s)",
                        n_replaced,
                        offload_fp8_weights,
                    )
                    if offload_fp8_weights and load_device.type == "cuda":
                        torch.cuda.empty_cache()
                    module._has_fp8_model = True

                logger.info("Custom Gemma weights loaded.")

                # DO NOT cast to torch_dtype here - that would upcast quantized weights to full precision!
                # module.model = module.model.to(dtype=torch_dtype)
            else:
                module.model = Gemma3ForConditionalGeneration.from_pretrained(
                    gemma_path,
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                )
        module._gemma_root = module._gemma_root or gemma_root

        # Ensure model is in eval mode
        if module.model is not None:
            module.model.eval()

        return module

    def load_tokenizer(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        tokenizer_max_length = int(os.getenv("LTX2_GEMMA_MAX_LENGTH", "1024"))
        if tokenizer_max_length <= 0:
            raise ValueError(f"LTX2_GEMMA_MAX_LENGTH must be > 0, got {tokenizer_max_length}")
        logger.info("Using Gemma tokenizer max_length=%d", tokenizer_max_length)
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_path, tokenizer_max_length)
        module._gemma_root = module._gemma_root or gemma_root
        return module

    gemma_load_ops = ModuleOps(
        "GemmaLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.model is None,
        mutator=load_gemma,
    )
    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    return (gemma_load_ops, tokenizer_load_ops)


def encode_text(text_encoder: GemmaTextEncoderModelBase, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Encode a list of prompts using the provided Gemma text encoder.
    Args:
        text_encoder: The Gemma text encoder instance.
        prompts: List of prompt strings to encode.
    Returns:
        List of tuples, each containing (v_context, a_context) tensors for each prompt.
    """
    result = []
    for prompt in prompts:
        v_context, a_context, _ = text_encoder(prompt)
        result.append((v_context, a_context))
    return result


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility.
    Flash Attention within SDPA requires sequence lengths aligned to 8 bytes.
    This pads input_ids, attention_mask, and token_type_ids (if present) to prevent
    'p.attn_bias_ptr is not correctly aligned' errors.
    """
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)

        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)

        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs
