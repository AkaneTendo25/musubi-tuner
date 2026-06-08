"""Inference-only HPSv3RewardInferencer.

Vendored / slimmed from github.com/MizzenAI/HPSv3 (hpsv3/inference.py +
hpsv3/train.py::create_model_and_processor). Reproduces the external inferencer's
behavior for non-quantized, non-LoRA inference while dropping the training-only
deps (fire, trl, peft, omegaconf, datasets, matplotlib, tensorboard, deepspeed).

Kept verbatim from the source: the model+processor build, the checkpoint load with
the transformers>=4.5x key-remap try/except, prepare_batch, and reward (logits[i][0]
is mu, higher=better). The trl helpers are inlined: HPSv3 is not quantized, so
get_quantization_config -> None and get_kbit_device_map is unused. HPSv3_7B.yaml
values are hardcoded (output_dim=2, reward_token/use_special_tokens via 'special',
rm_head_type='ranknet', rm_head_kwargs=None, torch_dtype=bfloat16,
disable_flash_attn2=False).
"""

from collections.abc import Mapping

import torch
from transformers import AutoProcessor

from .model import Qwen2VLRewardModelBT
from .templates import INSTRUCTION, prompt_with_special_token, prompt_without_special_token
from .vision_process import process_vision_info

try:
    import flash_attn  # noqa: F401

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

# HPSv3_7B.yaml values the original loader reads (verified against config/HPSv3_7B.yaml).
_DEFAULT_MODEL_NAME_OR_PATH = "Qwen/Qwen2-VL-7B-Instruct"
_OUTPUT_DIM = 2
_REWARD_TOKEN = "special"
_USE_SPECIAL_TOKENS = True
_RM_HEAD_TYPE = "ranknet"
_RM_HEAD_KWARGS = None
_TORCH_DTYPE = torch.bfloat16
_DISABLE_FLASH_ATTN2 = False


def _build_model_and_processor(model_name_or_path=_DEFAULT_MODEL_NAME_OR_PATH, cache_dir=None):
    """Inlined, inference-only replacement for hpsv3.train.create_model_and_processor.

    Non-quantized, non-LoRA path only: get_quantization_config(...) -> None so
    device_map / quantization_config stay None and get_kbit_device_map is unused.
    """
    processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="right", cache_dir=cache_dir)

    special_token_ids = None
    if _USE_SPECIAL_TOKENS:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_name_or_path,
        output_dim=_OUTPUT_DIM,
        reward_token=_REWARD_TOKEN,
        special_token_ids=special_token_ids,
        torch_dtype=_TORCH_DTYPE,
        attn_implementation=("flash_attention_2" if not _DISABLE_FLASH_ATTN2 and _HAS_FLASH_ATTN else "sdpa"),
        cache_dir=cache_dir,
        rm_head_type=_RM_HEAD_TYPE,
        rm_head_kwargs=_RM_HEAD_KWARGS,
        revision="main",
        device_map=None,
        quantization_config=None,
        use_cache=False,
    )

    if _USE_SPECIAL_TOKENS:
        model.resize_token_embeddings(len(processor.tokenizer))

    # training_args.bf16 == True for HPSv3_7B.yaml -> cast to bf16, keep rm_head fp32.
    model.to(torch.bfloat16)
    model.rm_head.to(torch.float32)

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor


class HPSv3RewardInferencer:
    def __init__(self, config_path=None, checkpoint_path=None, device="cuda", model_name_or_path=None, cache_dir=None):
        # config_path accepted for signature compat with the external inferencer; the
        # HPSv3_7B.yaml values are hardcoded above (no yaml parser).
        if checkpoint_path is None:
            import huggingface_hub

            checkpoint_path = huggingface_hub.hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model")

        model, processor = _build_model_and_processor(
            model_name_or_path=model_name_or_path or _DEFAULT_MODEL_NAME_OR_PATH,
            cache_dir=cache_dir,
        )

        self.device = device
        self.use_special_tokens = _USE_SPECIAL_TOKENS

        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # transformers>=4.5x renamed the Qwen2-VL submodules: the vision tower moved under
            # ``model.visual.*`` and the language model under ``model.language_model.*``. Older
            # HPSv3 checkpoints store them as ``visual.*`` + ``model.*``; remap to the new naming.
            def _remap_qwen2vl_key(k):
                if k.startswith("visual."):
                    return "model." + k
                if k.startswith("model."):
                    return "model.language_model." + k[len("model.") :]
                return k

            state_dict = {_remap_qwen2vl_key(k): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, "constant", self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, "constant", 0)

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text) + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def reward(self, prompts, image_paths):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]

        return rewards
