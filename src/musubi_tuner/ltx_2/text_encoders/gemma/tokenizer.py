from transformers import AutoTokenizer


class _SentencePieceTokenizerAdapter:
    """Thin adapter around sentencepiece.SentencePieceProcessor that exposes
    the subset of the HuggingFace tokenizer API used by LTXVGemmaTokenizer.
    Used when loading the tokenizer from raw spiece_model bytes extracted from
    a standalone safetensors file (no gemma_root directory).
    """

    def __init__(self, model_proto: bytes, max_length: int = 256):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromSerializedProto(model_proto)
        self.model_max_length = max_length
        self.padding_side = "left"
        self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.sp.eos_id()
        self.eos_token_id = self.sp.eos_id()
        self.pad_token = self.sp.IdToPiece(self.pad_token_id) if self.pad_token_id >= 0 else "<eos>"
        self.eos_token = self.sp.IdToPiece(self.eos_token_id) if self.eos_token_id >= 0 else "<eos>"

    def __call__(self, text, padding=None, max_length=None, truncation=False, return_tensors=None):
        import torch

        ids = self.sp.Encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]

        attn = [1] * len(ids)

        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                if self.padding_side == "left":
                    ids = [self.pad_token_id] * pad_len + ids
                    attn = [0] * pad_len + attn
                else:
                    ids = ids + [self.pad_token_id] * pad_len
                    attn = attn + [0] * pad_len

        if return_tensors == "pt":
            import torch

            class _Out:
                pass

            out = _Out()
            out.input_ids = torch.tensor([ids], dtype=torch.long)
            out.attention_mask = torch.tensor([attn], dtype=torch.long)
            return out

        return {"input_ids": [ids], "attention_mask": [attn]}

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError(
            "Chat template is not available when using a standalone safetensors tokenizer. "
            "Prompt enhancement requires --gemma_root with a full HuggingFace model directory."
        )


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(self, tokenizer_path: str | bytes, max_length: int = 256):
        """
        Initialize the tokenizer.
        Args:
            tokenizer_path (str | bytes): Path to the pretrained tokenizer files or model directory,
                or raw spiece_model bytes extracted from a safetensors file.
            max_length (int, optional): Max sequence length for encoding. Defaults to 256.
        """
        if isinstance(tokenizer_path, bytes):
            self.tokenizer = _SentencePieceTokenizerAdapter(tokenizer_path, max_length)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, model_max_length=max_length
            )
            # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out
