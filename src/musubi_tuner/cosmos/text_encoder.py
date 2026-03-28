# Text encoder for Cosmos Predict 2.5
# Uses Qwen2.5-VL-7B-Instruct as the text encoder (Cosmos-Reason1 architecture)
# Outputs FULL_CONCAT of all hidden layers: 28 layers × 3584 = 100352 dim
#
# Original: https://github.com/nvidia-cosmos/cosmos-predict2.5
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0

from typing import List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

NUM_EMBEDDING_PADDING_TOKENS = 512
QWEN_HIDDEN_SIZE = 3584
QWEN_NUM_HIDDEN_LAYERS = 28
# FULL_CONCAT: all hidden layers (1..28) concatenated
FULL_CONCAT_DIM = QWEN_NUM_HIDDEN_LAYERS * QWEN_HIDDEN_SIZE  # 100352


def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Mean normalize: subtract mean, divide by std (per-token)."""
    return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)


class CosmosTextEncoder:
    """Cosmos text encoder using Qwen2.5-VL-7B-Instruct from HuggingFace transformers.

    Extracts all hidden state layers, mean-normalizes each, and concatenates them
    (FULL_CONCAT strategy) to produce 100352-dim embeddings per token.

    The DiT model's crossattn_proj (a learned linear layer in the checkpoint)
    projects these to 1024-dim for cross-attention.
    """

    def __init__(
        self,
        model_name_or_path: str = "nvidia/Cosmos-Reason1-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = NUM_EMBEDDING_PADDING_TOKENS,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length

        logger.info(f"Loading Qwen2.5-VL text encoder from {model_name_or_path}")

        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            output_hidden_states=True,
        )
        self.model.eval()
        logger.info("Qwen2.5-VL text encoder loaded")

    def encode(self, prompts: List[str]) -> List[torch.Tensor]:
        """Encode prompts to 100352-dim embeddings using FULL_CONCAT strategy.

        Args:
            prompts: list of text prompts

        Returns:
            list of tensors, each [seq_len, 100352]
        """
        results = []
        for prompt in prompts:
            # Format as chat template (matching Cosmos training)
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]

            text = self.tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # outputs.hidden_states: tuple of (embedding_layer, layer1, ..., layer28)
            # We want layers 1..28 (skip embedding layer 0)
            hidden_states = outputs.hidden_states  # tuple of [1, seq_len, 3584]

            # Mean-normalize each layer and concatenate (FULL_CONCAT strategy)
            normalized_layers = []
            for layer_idx in range(1, len(hidden_states)):
                normalized = mean_normalize(hidden_states[layer_idx])
                normalized_layers.append(normalized)

            # FULL_CONCAT: cat all layers along last dim
            # [1, seq_len, 28*3584] = [1, seq_len, 100352]
            text_embedding = torch.cat(normalized_layers, dim=-1)
            text_embedding = text_embedding.squeeze(0)  # [seq_len, 100352]

            results.append(text_embedding.to(self.dtype))

        return results

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(device)
        return self
