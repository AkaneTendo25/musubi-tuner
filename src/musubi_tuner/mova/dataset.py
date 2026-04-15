import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Dataset


VIDEO_KEYS = ("video_latents", "visual_latents", "latents_video")
AUDIO_KEYS = ("audio_latents", "latents_audio")
COND_KEYS = ("conditioning_latents", "image_latents", "latents_image", "y")
PROMPT_KEYS = ("prompt_embeds", "t5_embeds", "prompt_embeddings", "t5")
FPS_KEYS = ("video_fps", "fps", "frame_rate")


@dataclass
class CacheItem:
    path: str
    video_fps: Optional[float] = None


def _discover_cache_files(cache_dir: Optional[str], cache_glob: str, manifest: Optional[str]) -> List[CacheItem]:
    items: List[CacheItem] = []
    if manifest:
        with open(manifest, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                path = row.get("cache_path") or row.get("path") or row.get("file")
                if not path:
                    raise ValueError(f"Manifest line {line_no} is missing cache_path/path/file")
                if cache_dir and not os.path.isabs(path):
                    path = os.path.join(cache_dir, path)
                items.append(CacheItem(path=os.path.abspath(path), video_fps=row.get("video_fps")))
    else:
        if not cache_dir:
            raise ValueError("cache_dir is required when manifest is not provided")
        pattern = os.path.join(cache_dir, cache_glob)
        for path in sorted(glob.glob(pattern, recursive=True)):
            items.append(CacheItem(path=os.path.abspath(path)))

    if not items:
        raise ValueError("No MOVA cache files were found")
    return items


def _find_key(data: Dict[str, torch.Tensor], candidates: Iterable[str], *, required: bool = True):
    for key in candidates:
        if key in data:
            return data[key], key
    if required:
        raise KeyError(f"Missing required tensor. Tried keys: {', '.join(candidates)}")
    return None, None


def _squeeze_optional_batch(tensor: torch.Tensor, expected_batched_ndim: int) -> torch.Tensor:
    if tensor.ndim == expected_batched_ndim and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _read_metadata_value(path: str, keys: Iterable[str]) -> Optional[float]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return None
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                return None
    return None


class MovaLatentDataset(Dataset):
    def __init__(self, cache_dir: Optional[str], cache_glob: str = "*.safetensors", manifest: Optional[str] = None):
        self.items = _discover_cache_files(cache_dir, cache_glob, manifest)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        tensors = load_file(item.path)

        video_latents, _ = _find_key(tensors, VIDEO_KEYS)
        audio_latents, _ = _find_key(tensors, AUDIO_KEYS)
        conditioning_latents, _ = _find_key(tensors, COND_KEYS)
        prompt_embeds, _ = _find_key(tensors, PROMPT_KEYS)
        fps_tensor, _ = _find_key(tensors, FPS_KEYS, required=False)

        video_latents = _squeeze_optional_batch(video_latents, 5).float()
        audio_latents = _squeeze_optional_batch(audio_latents, 3).float()
        conditioning_latents = _squeeze_optional_batch(conditioning_latents, 5).float()
        prompt_embeds = _squeeze_optional_batch(prompt_embeds, 3).float()

        if prompt_embeds.ndim != 2:
            raise ValueError(f"Prompt embeddings in {item.path} must have shape [seq, dim], got {tuple(prompt_embeds.shape)}")
        if video_latents.ndim != 4:
            raise ValueError(f"Video latents in {item.path} must have shape [C, T, H, W], got {tuple(video_latents.shape)}")
        if audio_latents.ndim != 2:
            raise ValueError(f"Audio latents in {item.path} must have shape [C, L], got {tuple(audio_latents.shape)}")
        if conditioning_latents.ndim != 4:
            raise ValueError(
                f"Conditioning latents in {item.path} must have shape [C, T, H, W], got {tuple(conditioning_latents.shape)}"
            )

        if fps_tensor is not None:
            video_fps = float(fps_tensor.reshape(-1)[0].item())
        elif item.video_fps is not None:
            video_fps = float(item.video_fps)
        else:
            video_fps = _read_metadata_value(item.path, FPS_KEYS) or 24.0

        return {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
            "conditioning_latents": conditioning_latents,
            "prompt_embeds": prompt_embeds,
            "video_fps": torch.tensor(video_fps, dtype=torch.float32),
            "cache_path": item.path,
        }


def mova_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not examples:
        raise ValueError("Empty MOVA batch")

    ref_video_shape = tuple(examples[0]["video_latents"].shape)
    ref_audio_shape = tuple(examples[0]["audio_latents"].shape)
    ref_cond_shape = tuple(examples[0]["conditioning_latents"].shape)

    for example in examples[1:]:
        if tuple(example["video_latents"].shape) != ref_video_shape:
            raise ValueError("All video latents in a batch must have the same shape")
        if tuple(example["audio_latents"].shape) != ref_audio_shape:
            raise ValueError("All audio latents in a batch must have the same shape")
        if tuple(example["conditioning_latents"].shape) != ref_cond_shape:
            raise ValueError("All conditioning latents in a batch must have the same shape")

    max_prompt_len = max(example["prompt_embeds"].shape[0] for example in examples)
    prompt_dim = examples[0]["prompt_embeds"].shape[1]
    padded_prompts = []
    for example in examples:
        prompt = example["prompt_embeds"]
        pad_len = max_prompt_len - prompt.shape[0]
        if pad_len > 0:
            prompt = F.pad(prompt, (0, 0, 0, pad_len))
        padded_prompts.append(prompt)

    return {
        "video_latents": torch.stack([example["video_latents"] for example in examples], dim=0),
        "audio_latents": torch.stack([example["audio_latents"] for example in examples], dim=0),
        "conditioning_latents": torch.stack([example["conditioning_latents"] for example in examples], dim=0),
        "prompt_embeds": torch.stack(padded_prompts, dim=0).view(len(examples), max_prompt_len, prompt_dim),
        "video_fps": torch.stack([example["video_fps"] for example in examples], dim=0),
        "cache_path": [example["cache_path"] for example in examples],
    }
