from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.t5gemma import T5GemmaEncoderModel

from ...common import CPUOffloadWrapper, get_arch_memory
from ...utils import env_is_true


def ensure_hf_safetensors_index(model_path: str) -> None:
    model_dir = Path(model_path)
    single_file = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"
    if single_file.exists() or index_file.exists():
        return

    shard_paths = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shard_paths:
        return

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_path in shard_paths:
        total_size += shard_path.stat().st_size
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard_file:
            for key in shard_file.keys():
                weight_map[key] = shard_path.name

    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_file.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")


class T5GemmaEncoder:
    def __init__(
        self,
        model_path: str,
        device: str,
        weight_dtype: torch.dtype,
        *,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: torch.dtype | None = None,
    ):
        self.device = device
        ensure_hf_safetensors_index(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be enabled for T5-Gemma.")

        if load_in_8bit or load_in_4bit:
            if not torch.cuda.is_available():
                raise ValueError("8-bit/4-bit T5-Gemma loading requires CUDA.")
            if device != "cuda":
                raise ValueError("8-bit/4-bit T5-Gemma loading currently requires --device cuda.")

            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("bitsandbytes/transformers quantization support is required for 4-bit/8-bit T5-Gemma loading.") from exc

            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                compute_dtype = bnb_4bit_compute_dtype if bnb_4bit_compute_dtype is not None else weight_dtype
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

            self.model = T5GemmaEncoderModel.from_pretrained(
                model_path,
                is_encoder_decoder=False,
                local_files_only=True,
                torch_dtype=weight_dtype,
                quantization_config=quantization_config,
                device_map={"": "cuda"},
            )
        else:
            model = T5GemmaEncoderModel.from_pretrained(
                model_path,
                is_encoder_decoder=False,
                local_files_only=True,
                torch_dtype=weight_dtype,
            ).to(device)
            self.model = CPUOffloadWrapper(model, is_cpu_offload=env_is_true("CPU_OFFLOAD") or get_arch_memory() <= 48)

    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"].half()


_t5_gemma_cache: Optional[T5GemmaEncoder] = None
_t5_gemma_cache_key: Optional[tuple] = None


def get_t5_gemma_encoder(
    model_path: str,
    device: str,
    weight_dtype: torch.dtype,
    *,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: torch.dtype | None = None,
) -> T5GemmaEncoder:
    global _t5_gemma_cache, _t5_gemma_cache_key
    cache_key = (
        model_path,
        device,
        weight_dtype,
        load_in_8bit,
        load_in_4bit,
        bnb_4bit_quant_type,
        bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype,
    )
    if _t5_gemma_cache is None or _t5_gemma_cache_key != cache_key:
        _t5_gemma_cache = T5GemmaEncoder(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
        _t5_gemma_cache_key = cache_key
    return _t5_gemma_cache


@torch.inference_mode()
def get_t5_gemma_embedding(
    prompt: str,
    model_path: str,
    device: str,
    weight_dtype: torch.dtype,
    *,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    encoder = get_t5_gemma_encoder(
        model_path=model_path,
        device=device,
        weight_dtype=weight_dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    return encoder.encode(prompt)
