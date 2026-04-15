from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(
    tokenizer_path: str,
    subfolder: Optional[str] = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer_path, subfolder=subfolder, trust_remote_code=trust_remote_code)


def load_text_encoder(
    model_path: str,
    dtype: torch.dtype,
    subfolder: Optional[str] = None,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    return AutoModel.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def encode_hidden_states(text_encoder: PreTrainedModel, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if getattr(text_encoder.config, "is_encoder_decoder", False) and hasattr(text_encoder, "encoder"):
        outputs = text_encoder.encoder(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs.get("attention_mask"),
            return_dict=True,
        )
    else:
        outputs = text_encoder(**encoded_inputs, return_dict=True)

    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = outputs[0]
    return hidden_states
