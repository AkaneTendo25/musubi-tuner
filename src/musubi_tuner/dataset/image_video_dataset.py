from concurrent.futures import ThreadPoolExecutor
import glob
from importlib.util import find_spec
import json
import math
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import numpy as np
import torch
from safetensors.torch import save_file, load_file
from PIL import Image
import cv2
import av

from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF"]

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"]


if find_spec("jxlpy") is not None:  # JPEG-XL on Linux
    from jxlpy import JXLImagePlugin  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if find_spec("pillow_jxl") is not None:  # JPEG-XL on Windows
    import pillow_jxl  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".MP4",
    ".WEBM",
    ".AVI",
    ".MKV",
    ".MOV",
    ".FLV",
    ".WMV",
    ".M4V",
    ".MPG",
    ".MPEG",
]  # some of them are not tested
MASK_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
MASK_METADATA_EXTENSIONS = [".json", ".JSON", ".txt", ".TXT", ".csv", ".CSV"]

# Architecture short names cannot contain underscore
ARCHITECTURE_HUNYUAN_VIDEO = "hv"
ARCHITECTURE_HUNYUAN_VIDEO_FULL = "hunyuan_video"
ARCHITECTURE_WAN = "wan"
ARCHITECTURE_WAN_FULL = "wan"
ARCHITECTURE_LTX2 = "ltx2"
ARCHITECTURE_LTX2_FULL = "ltx2_v1"
ARCHITECTURE_FRAMEPACK = "fp"
ARCHITECTURE_FRAMEPACK_FULL = "framepack"
ARCHITECTURE_FLUX_KONTEXT = "fk"
ARCHITECTURE_FLUX_KONTEXT_FULL = "flux_kontext"
ARCHITECTURE_FLUX_2_DEV = "f2d"
ARCHITECTURE_FLUX_2_DEV_FULL = "flux_2_dev"
ARCHITECTURE_FLUX_2_KLEIN_4B = "f2k4b"
ARCHITECTURE_FLUX_2_KLEIN_4B_FULL = "flux_2_klein_4b"
ARCHITECTURE_FLUX_2_KLEIN_9B = "f2k9b"
ARCHITECTURE_FLUX_2_KLEIN_9B_FULL = "flux_2_klein_9b"
ARCHITECTURE_QWEN_IMAGE = "qi"
ARCHITECTURE_QWEN_IMAGE_FULL = "qwen_image"
ARCHITECTURE_QWEN_IMAGE_EDIT = "qie"
ARCHITECTURE_QWEN_IMAGE_EDIT_FULL = "qwen_image_edit"
ARCHITECTURE_QWEN_IMAGE_LAYERED = "qil"
ARCHITECTURE_QWEN_IMAGE_LAYERED_FULL = "qwen_image_layered"
ARCHITECTURE_KANDINSKY5 = "k5"
ARCHITECTURE_KANDINSKY5_FULL = "kandinsky5"
ARCHITECTURE_HUNYUAN_VIDEO_1_5 = "hv15"
ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL = "hunyuan_video_1_5"
ARCHITECTURE_Z_IMAGE = "zi"
ARCHITECTURE_Z_IMAGE_FULL = "z_image"


def glob_images(directory, base="*", caption_extension=None):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates

    # check for caption files and only keep images with captions
    if caption_extension is not None:
        caption_paths = glob.glob(os.path.join(glob.escape(directory), "*" + caption_extension))
        caption_bases = set()
        for caption_path in caption_paths:
            caption_name = os.path.basename(caption_path)
            if caption_name.endswith(caption_extension):
                caption_base = caption_name[: -len(caption_extension)]
            else:
                caption_base = os.path.splitext(caption_name)[0]
            caption_bases.add(caption_base)
        filtered_img_paths = []
        for img_path in img_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            if img_base in caption_bases:
                filtered_img_paths.append(img_path)
        img_paths = filtered_img_paths

    img_paths.sort()
    return img_paths


def glob_audio(directory, base="*"):
    audio_paths = []
    for ext in AUDIO_EXTENSIONS:
        if base == "*":
            audio_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            audio_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    audio_paths = list(set(audio_paths))  # remove duplicates
    audio_paths.sort()
    return audio_paths


def glob_videos(directory, base="*"):
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        if base == "*":
            video_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            video_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    video_paths = list(set(video_paths))  # remove duplicates
    video_paths.sort()
    return video_paths


def find_stem_matched_file(directory: Optional[str], stem: str, extensions: Optional[Sequence[str]] = None) -> Optional[str]:
    if directory is None:
        return None
    extensions = extensions or MASK_EXTENSIONS
    for ext in extensions:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    candidate_dir = os.path.join(directory, stem)
    if os.path.isdir(candidate_dir):
        return candidate_dir
    return None


def load_loss_mask_image(mask_path: str, *, invert: bool = False) -> Image.Image:
    mask = Image.open(mask_path)
    if "A" in mask.getbands():
        mask = mask.getchannel("A")
    else:
        mask = mask.convert("L")
    if invert:
        from PIL import ImageOps

        mask = ImageOps.invert(mask)
    return mask


def alpha_channel_to_loss_mask(image: Image.Image, *, invert: bool = False) -> Optional[Image.Image]:
    if "A" not in image.getbands():
        return None
    mask = image.getchannel("A")
    if invert:
        from PIL import ImageOps

        mask = ImageOps.invert(mask)
    return mask


def loss_mask_to_float_array(mask: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Image.Image) and mask.mode != "L":
        mask = mask.convert("L")
    arr = resize_image_to_bucket(mask, bucket_reso)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0


def load_loss_mask_frames(
    mask_path: str,
    *,
    bucket_reso: tuple[int, int],
    frame_count: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
    invert: bool = False,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive for loss mask loading, got {frame_count}")

    if os.path.isfile(mask_path) and os.path.splitext(mask_path)[1] in IMAGE_EXTENSIONS:
        mask = load_loss_mask_image(mask_path, invert=invert)
        mask_frames = [loss_mask_to_float_array(mask, bucket_reso)] * frame_count
    else:
        frames = load_video(
            mask_path,
            start_frame=start_frame,
            end_frame=end_frame,
            bucket_reso=bucket_reso,
            source_fps=source_fps,
            target_fps=target_fps,
        )
        if not frames:
            raise ValueError(f"No frames decoded from loss mask path: {mask_path}")

        mask_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame
            if image.mode != "L":
                image = image.convert("L")
            if invert:
                from PIL import ImageOps

                image = ImageOps.invert(image)
            mask_frames.append(loss_mask_to_float_array(image, bucket_reso))

        if len(mask_frames) < frame_count:
            mask_frames.extend([mask_frames[-1]] * (frame_count - len(mask_frames)))
        elif len(mask_frames) > frame_count:
            mask_frames = mask_frames[:frame_count]

    return np.stack(mask_frames, axis=0).astype(np.float32)


def load_audio_loss_mask_intervals(mask_path: str) -> Optional[list[tuple[float, float]]]:
    ext = os.path.splitext(mask_path)[1].lower()
    if ext == ".json":
        with open(mask_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("loss_mask_intervals", data.get("audio_loss_mask_intervals", data.get("intervals")))
        return normalize_loss_mask_intervals(data)

    intervals: list[tuple[float, float]] = []
    with open(mask_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p for p in stripped.replace(",", " ").split() if p]
            if len(parts) < 2:
                raise ValueError(f"Audio loss mask interval line must contain start and end seconds: {line!r}")
            intervals.append((float(parts[0]), float(parts[1])))
    return intervals


def normalize_loss_mask_intervals(value: Any) -> Optional[list[tuple[float, float]]]:
    if value is None:
        return None
    intervals: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start", item.get("start_time", item.get("from")))
            end = item.get("end", item.get("end_time", item.get("to")))
        else:
            start, end = item[0], item[1]
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            raise ValueError(f"Invalid loss mask interval with end <= start: {(start_f, end_f)}")
        intervals.append((start_f, end_f))
    return intervals


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def _normalize_optional_path_list(
    primary: Optional[str] = None,
    extras: Optional[Sequence[str]] = None,
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for value in ([primary] if primary is not None else []) + list(extras or []):
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)

    return values


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


class ItemInfo:
    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: tuple[int, int],
        bucket_size: Optional[tuple[Any]] = None,
        frame_count: Optional[int] = None,
        content: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        latent_cache_path: Optional[str] = None,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.frame_count = frame_count
        self.content = content
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None
        self.reference_latent_cache_path: Optional[str] = None
        self.reference_audio_latent_cache_path: Optional[str] = None
        self.reference_latent_cache_paths: Optional[list[str]] = None
        self.reference_audio_latent_cache_paths: Optional[list[str]] = None

        # np.ndarray for video, list[np.ndarray] for image with multiple controls
        self.control_content: Optional[Union[np.ndarray, list[np.ndarray]]] = None
        self.loss_mask_content: Optional[np.ndarray] = None
        self.loss_mask_path: Optional[str] = None
        self.audio_loss_mask_intervals: Optional[list[tuple[float, float]]] = None

        # FramePack architecture specific
        self.fp_latent_window_size: Optional[int] = None
        self.fp_1f_clean_indices: Optional[list[int]] = None  # indices of clean latents for 1f
        self.fp_1f_target_index: Optional[int] = None  # target index for 1f clean latents
        self.fp_1f_no_post: Optional[bool] = None  # whether to add zero values as clean latent post

    def __str__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            + f"frame_count={self.frame_count}, latent_cache_path={self.latent_cache_path}, "
            + f"content={[c.shape for c in self.content] if isinstance(self.content, list) else (self.content.shape if self.content is not None else None)}), "
            + f"control_content={[cc.shape for cc in self.control_content] if isinstance(self.control_content, list) else (self.control_content.shape if self.control_content is not None else None)})"
        )


def select_caption_from_metadata(data: dict[str, Any], caption_field: Optional[str] = None) -> str:
    field = caption_field or "caption"
    if field not in data:
        raise KeyError(
            f"Caption field {field!r} was not found in metadata item. "
            f"Available keys: {sorted(data.keys())}"
        )
    caption = data[field]
    if caption is None:
        return ""
    if not isinstance(caption, str):
        raise TypeError(f"Caption field {field!r} must be a string, got {type(caption).__name__}")
    return caption


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
):
    """Wan architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_ltx2(item_info: ItemInfo, latent: torch.Tensor, extra_tensors: Optional[dict[str, torch.Tensor]] = None):
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}
    if extra_tensors:
        for key, value in extra_tensors.items():
            sd[key] = value.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_LTX2_FULL)


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
):
    """FramePack architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()

    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_flux_kontext(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: torch.Tensor,
):
    """FLUX.1 Kontext architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    _, H, W = control_latent.shape
    F = 1
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_latent_cache_flux_2(
    item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]], arch_full: str
):
    """Flux 2 architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, arch_full)


def save_latent_cache_qwen_image(item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]]):
    """Qwen-Image architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"
    assert control_latent is None or all(cl.dim() == 4 for cl in control_latent), (
        "control_latent should be 4D tensor (frame, channel, height, width) or None"
    )

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, F, H, W = cl.shape
            sd[f"latents_control_{i}_{F}x{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_latent_cache_kandinsky5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor] = None,
    control_latent: Optional[torch.Tensor] = None,
    scaling_factor: Optional[float] = None,
):
    """Kandinsky 5 architecture (image/video), with optional source/control latents for i2v/control."""
    assert latent.dim() == 3 or latent.dim() == 4, "latent should be 3D (C,H,W) or 4D (F,C,H,W) tensor"

    if latent.dim() == 4:
        _, F, H, W = latent.shape
    else:
        F, H, W = 1, latent.shape[1], latent.shape[2]
        latent = latent.unsqueeze(0)
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous().clone()}

    if image_latent is not None:
        _, F_img, H_img, W_img = image_latent.shape
        sd[f"latents_image_{F_img}x{H_img}x{W_img}_{dtype_str}"] = image_latent.detach().cpu().contiguous().clone()

    if control_latent is not None:
        _, F_ctrl, H_ctrl, W_ctrl = control_latent.shape
        sd[f"latents_control_{F_ctrl}x{H_ctrl}x{W_ctrl}_{dtype_str}"] = control_latent.detach().cpu().contiguous().clone()

    if scaling_factor is not None:
        sd["vae_scaling_factor"] = torch.tensor(float(scaling_factor))

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_latent_cache_hunyuan_video_1_5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor],
    vision_feature: Optional[torch.Tensor],
):
    """HunyuanVideo 1.5 architecture"""
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd: dict[str, torch.Tensor] = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if image_latent is not None:
        dtype_str = dtype_to_str(image_latent.dtype)
        _, F, H, W = image_latent.shape
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if vision_feature is not None:
        dtype_str = dtype_to_str(vision_feature.dtype)
        sd[f"siglip_{dtype_str}"] = vision_feature.detach().cpu()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_latent_cache_z_image(item_info: ItemInfo, latent: torch.Tensor):
    """Z-Image architecture. No control latent is supported."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    C, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_latent_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    metadata = {
        "architecture": arch_fullname,
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture"""
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_ltx2(item_info: ItemInfo, embed: torch.Tensor, mask: torch.Tensor):
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"text_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd["text_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_LTX2_FULL)


def save_text_encoder_output_cache_ltx2_gemma(
    item_info: ItemInfo,
    *,
    video_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    audio_prompt_embeds: Optional[torch.Tensor] = None,
    video_features: Optional[torch.Tensor] = None,
    audio_features: Optional[torch.Tensor] = None,
):
    assert video_prompt_embeds.dim() == 1 or video_prompt_embeds.dim() == 2, (
        f"video_prompt_embeds should be 2D tensor (feature, hidden_size) or (hidden_size,), got {video_prompt_embeds.shape}"
    )
    assert prompt_attention_mask is None or prompt_attention_mask.dim() == 1, (
        f"prompt_attention_mask should be 1D tensor (feature), got {prompt_attention_mask.shape}"
    )
    if audio_prompt_embeds is not None:
        assert audio_prompt_embeds.dim() == 1 or audio_prompt_embeds.dim() == 2, (
            f"audio_prompt_embeds should be 2D tensor (feature, hidden_size) or (hidden_size,), got {audio_prompt_embeds.shape}"
        )

    sd = {}
    dtype_str = dtype_to_str(video_prompt_embeds.dtype)

    sd[f"video_prompt_embeds_{dtype_str}"] = video_prompt_embeds.detach().cpu()
    if audio_prompt_embeds is not None:
        sd[f"audio_prompt_embeds_{dtype_str}"] = audio_prompt_embeds.detach().cpu()
    if prompt_attention_mask is not None:
        sd["prompt_attention_mask"] = prompt_attention_mask.detach().cpu()

    # Pre-connector features for --train_connectors training
    if video_features is not None:
        sd[f"video_features_{dtype_str}"] = video_features.detach().cpu()
    if audio_features is not None:
        sd[f"audio_features_{dtype_str}"] = audio_features.detach().cpu()

    text = video_prompt_embeds
    if audio_prompt_embeds is not None:
        text = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
    sd[f"text_{dtype_str}"] = text.detach().cpu()
    if prompt_attention_mask is not None:
        sd["text_mask"] = prompt_attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_LTX2_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd["llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_flux_kontext(item_info: ItemInfo, t5_vec: torch.Tensor, clip_l_pooler: torch.Tensor):
    """Flux Kontext architecture."""

    sd = {}
    dtype_str = dtype_to_str(t5_vec.dtype)
    sd[f"t5_vec_{dtype_str}"] = t5_vec.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str):
    """Flux 2 architecture."""

    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_qwen_image(item_info: ItemInfo, embed: torch.Tensor):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_text_encoder_output_cache_kandinsky5(
    item_info: ItemInfo, text_embeds: torch.Tensor, pooled_embed: torch.Tensor, attention_mask: torch.Tensor
):
    """Kandinsky 5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(text_embeds.dtype)
    sd[f"text_embeds_{dtype_str}"] = text_embeds.detach().cpu()
    dtype_str = dtype_to_str(pooled_embed.dtype)
    sd[f"pooled_embed_{dtype_str}"] = pooled_embed.detach().cpu()
    sd["attention_mask"] = attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_text_encoder_output_cache_hunyuan_video_1_5(item_info: ItemInfo, embed: torch.Tensor, byt5_embed: torch.Tensor):
    """Hunyuan-Video 1.5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()
    dtype_str = dtype_to_str(byt5_embed.dtype)
    sd[f"varlen_byt5_embed_{dtype_str}"] = byt5_embed.detach().cpu()
    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_text_encoder_output_cache_z_image(item_info: ItemInfo, embed: torch.Tensor):
    """Z-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_llm_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_text_encoder_output_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": arch_fullname,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if key not in sd:  # avoid overwriting by existing cache, we keep the new one
                    sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


class BucketSelector:
    RESOLUTION_STEPS_HUNYUAN = 16
    RESOLUTION_STEPS_WAN = 16
    RESOLUTION_STEPS_LTX2 = 32
    RESOLUTION_STEPS_FRAMEPACK = 16
    RESOLUTION_STEPS_FLUX_KONTEXT = 16
    RESOLUTION_STEPS_FLUX_2 = 16
    RESOLUTION_STEPS_QWEN_IMAGE = 16
    RESOLUTION_STEPS_QWEN_IMAGE_EDIT = 16
    RESOLUTION_STEPS_KANDINSKY5 = 16
    RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5 = 16
    RESOLUTION_STEPS_Z_IMAGE = 16

    ARCHITECTURE_STEPS_MAP = {
        ARCHITECTURE_HUNYUAN_VIDEO: RESOLUTION_STEPS_HUNYUAN,
        ARCHITECTURE_WAN: RESOLUTION_STEPS_WAN,
        ARCHITECTURE_LTX2: RESOLUTION_STEPS_LTX2,
        ARCHITECTURE_FRAMEPACK: RESOLUTION_STEPS_FRAMEPACK,
        ARCHITECTURE_FLUX_KONTEXT: RESOLUTION_STEPS_FLUX_KONTEXT,
        ARCHITECTURE_FLUX_2_DEV: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_4B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_9B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_QWEN_IMAGE: RESOLUTION_STEPS_QWEN_IMAGE,
        ARCHITECTURE_QWEN_IMAGE_EDIT: RESOLUTION_STEPS_QWEN_IMAGE_EDIT,
        ARCHITECTURE_QWEN_IMAGE_LAYERED: RESOLUTION_STEPS_QWEN_IMAGE,  # use same steps as Qwen-Image
        ARCHITECTURE_KANDINSKY5: RESOLUTION_STEPS_KANDINSKY5,
        ARCHITECTURE_HUNYUAN_VIDEO_1_5: RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5,
        ARCHITECTURE_Z_IMAGE: RESOLUTION_STEPS_Z_IMAGE,
    }

    @classmethod
    def resolve_resolution_steps(cls, architecture: str, reference_downscale: int = 1) -> int:
        if architecture not in BucketSelector.ARCHITECTURE_STEPS_MAP:
            raise ValueError(f"Invalid architecture: {architecture}")

        reso_steps = BucketSelector.ARCHITECTURE_STEPS_MAP[architecture]
        reference_downscale = max(1, int(reference_downscale or 1))
        if architecture == ARCHITECTURE_LTX2 and reference_downscale > 1:
            # LTX2 reference latents are quantized to /32 after spatial downscale.
            # Make target buckets divisible by 32 * downscale to avoid lossy flooring.
            reso_steps *= reference_downscale
        return reso_steps

    def __init__(
        self,
        resolution: Tuple[int, int],
        enable_bucket: bool = True,
        no_upscale: bool = False,
        architecture: str = "no_default",
        reference_downscale: int = 1,
    ):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.architecture = architecture

        self.reso_steps = BucketSelector.resolve_resolution_steps(architecture, reference_downscale)

        if not enable_bucket:
            # only define one bucket
            if resolution[0] % self.reso_steps != 0 or resolution[1] % self.reso_steps != 0:
                raise ValueError(f"resolution must be divisible by {self.reso_steps} for architecture={architecture}: {resolution}")
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: tuple[int, int],
        resolution: tuple[int, int],
        reso_steps: Optional[int] = None,
        architecture: Optional[str] = None,
        reference_downscale: int = 1,
    ) -> tuple[int, int]:
        """
        Get the bucket resolution for the given image size, resolution and resolution steps.
        Return (width, height).
        """
        if reso_steps is None and architecture is None:
            raise ValueError("resolution steps or architecture must be provided")
        if reso_steps is None and architecture is not None:
            reso_steps = BucketSelector.resolve_resolution_steps(architecture, reference_downscale)

        max_area = resolution[0] * resolution[1]
        width, height = image_size
        aspect_ratio = width / height
        bucket_width = int(math.sqrt(max_area * aspect_ratio))
        bucket_height = int(math.sqrt(max_area / aspect_ratio))
        bucket_width = divisible_by(bucket_width, reso_steps)
        bucket_height = divisible_by(bucket_height, reso_steps)

        # find appropriate resolutions
        best_resolution = None
        best_aspect_ratio_diff = float("inf")
        for i in range(-2, 3):
            w = bucket_width + i * reso_steps
            h = divisible_by(max_area // w, reso_steps)
            current_aspect_ratio_diff = abs((w / h) - aspect_ratio)
            if current_aspect_ratio_diff < best_aspect_ratio_diff:
                best_aspect_ratio_diff = current_aspect_ratio_diff
                best_resolution = (w, h)

        if best_resolution is not None:
            return best_resolution

        return bucket_width, bucket_height


def load_video(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    bucket_selector: Optional[BucketSelector] = None,
    bucket_reso: Optional[tuple[int, int]] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
) -> list[np.ndarray]:
    """
    bucket_reso: if given, resize the video to the bucket resolution, (width, height)
    """
    # auto-detect source FPS from video container when not explicitly set
    if source_fps is None and target_fps is not None and os.path.isfile(video_path):
        try:
            with av.open(video_path) as probe_container:
                stream = probe_container.streams.video[0]
                detected = stream.average_rate or stream.base_rate
                if detected and float(detected) > 0:
                    source_fps = float(detected)
                    # Keep this at debug level to avoid per-file log spam.
                    logger.debug(f"Auto-detected source FPS: {source_fps:.2f} for {os.path.basename(video_path)}")
        except Exception:
            pass  # detection failed, fall through to no-conversion branch

    # skip resampling when source and target FPS are nearly equal
    # ceil the source FPS so that e.g. 23.976 -> 24, then compare against target (25): diff=1, skip
    from musubi_tuner.ltx_2.env import get_ltx2_env

    fps_threshold = get_ltx2_env().fps_resampling_threshold
    needs_resampling = (
        source_fps is not None
        and target_fps is not None
        and abs(math.ceil(source_fps) - target_fps) > fps_threshold
    )

    if not needs_resampling and source_fps is not None and target_fps is not None and source_fps != target_fps:
        logger.info(
            f"Skipping FPS resampling for {os.path.basename(video_path)}: "
            f"source {source_fps:.3f} FPS within threshold of target {target_fps:.1f} FPS "
            f"(ceil={math.ceil(source_fps)}, diff={abs(math.ceil(source_fps) - target_fps)}, threshold={fps_threshold})"
        )

    if not needs_resampling:
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            for i in range(len(image_files)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)
    else:
        # drop frames to match the target fps TODO commonize this code with the above if this works
        logger.info(f"Resampling {os.path.basename(video_path)}: {source_fps:.2f} FPS -> {target_fps:.2f} FPS")
        frame_index_delta = target_fps / source_fps  # example: 16 / 30 = 0.5333
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i, frame in enumerate(container.decode(video=0)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i in range(len(image_files)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)

    return video


class BucketBatchManager:
    def __init__(
        self,
        bucketed_item_info: dict[tuple[Any], list[ItemInfo]],
        batch_size: int,
        num_timestep_buckets: Optional[int] = None,
        architecture: Optional[str] = None,
        target_fps: float = 24.0,
        audio_bucket_strategy: str = "pad",
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
    ):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()
        self.num_timestep_buckets = num_timestep_buckets
        self.timestep_pool = None
        self.architecture = architecture
        self.target_fps = target_fps
        self.audio_bucket_strategy = audio_bucket_strategy
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight

        # indices for enumerating batches. each batch is reso + batch_idx. reso is (width, height) or (width, height, frames)
        self.bucket_batch_indices: list[tuple[tuple[Any], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i))

        # do no shuffle here to avoid multiple datasets have different order
        # self.shuffle()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            logger.info(f"bucket: {bucket_reso}, count: {len(bucket)}")

        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        # shuffle each bucket
        for bucket in self.buckets.values():
            random.shuffle(bucket)

        # shuffle the order of batches
        random.shuffle(self.bucket_batch_indices)

        if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
            # prepare timesteps for each timestep buckets

            # 1. Calculate total number of timesteps needed for the entire epoch
            num_batches = len(self.bucket_batch_indices)
            total_timesteps_needed = num_batches * self.batch_size

            # 2. Generate a single large pool of stratified timesteps
            all_timesteps = []
            samples_per_bucket = math.ceil(total_timesteps_needed / self.num_timestep_buckets)

            for i in range(self.num_timestep_buckets):
                min_t = i / self.num_timestep_buckets
                max_t = (i + 1) / self.num_timestep_buckets
                for _ in range(samples_per_bucket):
                    all_timesteps.append(random.uniform(min_t, max_t))

            # 3. Shuffle the entire pool thoroughly
            random.shuffle(all_timesteps)

            # Trim the excess timesteps to match the exact number needed
            all_timesteps = all_timesteps[:total_timesteps_needed]

            # 4. Create the final timestep pool by chunking the shuffled list
            self.timestep_pool = []
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                self.timestep_pool.append(all_timesteps[start_idx:end_idx])
                # print(f"timestep pool {i}: {self.timestep_pool[-1]}")

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx):
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))
        batch_count = max(end - start, 0)

        batch_tensor_data = {}
        varlen_keys = set()

        audio_latents_per_item = []
        audio_lengths_per_item = []
        audio_loss_masks_per_item = []
        ref_audio_latents_per_item = []
        ref_audio_lengths_per_item = []
        dino_features_per_item = []
        diag_collect_keys = os.getenv("LTX2_NAN_DIAG", "0") == "1"
        item_keys = []
        latent_cache_paths = []
        audio_cache_paths = []
        text_cache_paths = []
        captions: list[str] = []
        for item_info in bucket[start:end]:
            sd_latent = load_file(item_info.latent_cache_path)
            audio_latent_cache_path = getattr(item_info, "audio_latent_cache_path", None)
            if audio_latent_cache_path is not None and os.path.exists(audio_latent_cache_path):
                sd_audio = load_file(audio_latent_cache_path)
                sd_latent = {**sd_latent, **sd_audio}

            dino_cache_path = getattr(item_info, "dino_feature_cache_path", None)
            if dino_cache_path is not None and os.path.exists(dino_cache_path):
                sd_dino = load_file(dino_cache_path)
                dino_features_per_item.append(sd_dino["dino_features"])  # [T_pixel, N_patches, D_dino]
            else:
                dino_features_per_item.append(None)

            reference_latent_cache_paths = getattr(item_info, "reference_latent_cache_paths", None)
            if not reference_latent_cache_paths:
                reference_latent_cache_path = getattr(item_info, "reference_latent_cache_path", None)
                if reference_latent_cache_path is not None:
                    reference_latent_cache_paths = [reference_latent_cache_path]
            if reference_latent_cache_paths:
                for ref_index, reference_latent_cache_path in enumerate(reference_latent_cache_paths):
                    if not os.path.exists(reference_latent_cache_path):
                        raise FileNotFoundError(f"Reference latent cache file not found: {reference_latent_cache_path}")
                    sd_ref = load_file(reference_latent_cache_path)
                    sd_ref_latents = {}
                    for key, value in sd_ref.items():
                        if key.startswith("latents_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("latents_", f"ref_latents_{ref_index}_", 1)
                            sd_ref_latents[mapped_key] = value
                    if not sd_ref_latents:
                        raise ValueError(f"No latent tensors found in reference cache: {reference_latent_cache_path}")
                    sd_latent = {**sd_latent, **sd_ref_latents}

            reference_audio_latent_cache_paths = getattr(item_info, "reference_audio_latent_cache_paths", None)
            if not reference_audio_latent_cache_paths:
                reference_audio_latent_cache_path = getattr(item_info, "reference_audio_latent_cache_path", None)
                if reference_audio_latent_cache_path is not None:
                    reference_audio_latent_cache_paths = [reference_audio_latent_cache_path]
            if reference_audio_latent_cache_paths:
                for ref_index, reference_audio_latent_cache_path in enumerate(reference_audio_latent_cache_paths):
                    if not os.path.exists(reference_audio_latent_cache_path):
                        raise FileNotFoundError(
                            f"Reference audio latent cache file not found: {reference_audio_latent_cache_path}"
                        )
                    sd_ref_audio_raw = load_file(reference_audio_latent_cache_path)
                    sd_ref_audio = {}
                    for key, value in sd_ref_audio_raw.items():
                        if key.startswith("audio_latents_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("audio_latents_", f"ref_audio_latents_{ref_index}_", 1)
                            sd_ref_audio[mapped_key] = value
                        elif key.startswith("audio_lengths_"):
                            if ref_index == 0:
                                mapped_key = "ref_" + key
                            else:
                                mapped_key = key.replace("audio_lengths_", f"ref_audio_lengths_{ref_index}_", 1)
                            sd_ref_audio[mapped_key] = value
                    if not sd_ref_audio:
                        raise ValueError(
                            f"No audio latent tensors found in reference audio cache: {reference_audio_latent_cache_path}"
                        )
                    sd_latent = {**sd_latent, **sd_ref_audio}

            sd_te = load_file(item_info.text_encoder_output_cache_path)
            sd = {**sd_latent, **sd_te}

            item_audio_latents = None
            item_audio_lengths = None
            item_audio_loss_mask = None
            item_ref_audio_latents: dict[int, torch.Tensor] = {}
            item_ref_audio_lengths: dict[int, torch.Tensor] = {}
            for key, value in sorted(sd.items()):
                if key.startswith("audio_latents_"):
                    item_audio_latents = value
                elif key.startswith("audio_lengths_"):
                    item_audio_lengths = value
                elif key == "audio_loss_mask":
                    item_audio_loss_mask = value
                elif key.startswith("ref_audio_latents_"):
                    ref_suffix = key[len("ref_audio_latents_") :]
                    ref_index = 0
                    if "_" in ref_suffix:
                        maybe_index, _rest = ref_suffix.split("_", 1)
                        if maybe_index.isdigit():
                            ref_index = int(maybe_index)
                    item_ref_audio_latents[ref_index] = value
                elif key.startswith("ref_audio_lengths_"):
                    ref_suffix = key[len("ref_audio_lengths_") :]
                    ref_index = 0
                    if "_" in ref_suffix:
                        maybe_index, _rest = ref_suffix.split("_", 1)
                        if maybe_index.isdigit():
                            ref_index = int(maybe_index)
                    item_ref_audio_lengths[ref_index] = value
            audio_latents_per_item.append(item_audio_latents)
            audio_lengths_per_item.append(item_audio_lengths)
            audio_loss_masks_per_item.append(item_audio_loss_mask)
            ref_audio_latents_per_item.append(
                [item_ref_audio_latents[idx] for idx in sorted(item_ref_audio_latents.keys())] if item_ref_audio_latents else None
            )
            ref_audio_lengths_per_item.append(
                [item_ref_audio_lengths[idx] for idx in sorted(item_ref_audio_lengths.keys())] if item_ref_audio_lengths else None
            )

            if diag_collect_keys:
                item_keys.append(item_info.item_key)
                latent_cache_paths.append(item_info.latent_cache_path)
                audio_cache_paths.append(audio_latent_cache_path)
                text_cache_paths.append(item_info.text_encoder_output_cache_path)
            captions.append(item_info.caption)

            # TODO refactor this
            for key in sd.keys():
                if (
                    key.startswith("audio_latents_")
                    or key.startswith("audio_lengths_")
                    or key == "audio_loss_mask"
                    or key.startswith("ref_audio_latents_")
                    or key.startswith("ref_audio_lengths_")
                ):
                    continue
                is_varlen_key = key.startswith("varlen_")  # varlen keys are not stacked
                content_key = key

                if is_varlen_key:
                    content_key = content_key.replace("varlen_", "")

                if content_key.endswith("_mask"):
                    pass
                else:
                    content_key = content_key.rsplit("_", 1)[0]  # remove dtype
                    if (
                        content_key.startswith("latents_")
                        or content_key.startswith("audio_latents_")
                        or content_key.startswith("ref_latents_")
                    ):
                        content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(sd[key])

                if is_varlen_key:
                    varlen_keys.add(content_key)

        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])

        if self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            present_audio = [x for x in audio_latents_per_item if isinstance(x, torch.Tensor)]
            if present_audio:
                ref = present_audio[0]
                if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
                    raise ValueError(
                        f"Expected cached audio latents to be 3D [C, T, F] before stacking, got: {getattr(ref, 'shape', None)}"
                    )

                channels = int(ref.shape[0])
                mel_bins = int(ref.shape[2])
                dtype = ref.dtype
                device = ref.device

                if self.audio_bucket_strategy == "truncate":
                    # Truncate mode: extract quantized_t from bucket_reso (last int element)
                    # and truncate all audio latents to that length — no padding needed.
                    quantized_t = None
                    for elem in reversed(bucket_reso):
                        if isinstance(elem, int):
                            quantized_t = elem
                            break
                    if quantized_t is None or quantized_t <= 0:
                        quantized_t = int(ref.shape[1])

                    truncated = []
                    truncated_masks = []
                    has_audio_loss_masks = any(isinstance(mask, torch.Tensor) for mask in audio_loss_masks_per_item)
                    for lat in audio_latents_per_item:
                        item_index = len(truncated)
                        if isinstance(lat, torch.Tensor):
                            if lat.dim() != 3:
                                raise ValueError(f"Expected audio latents to be 3D [C, T, F], got {tuple(lat.shape)}")
                            if int(lat.shape[0]) != channels or int(lat.shape[2]) != mel_bins:
                                raise ValueError(
                                    "Audio latents shape mismatch in batch: "
                                    f"expected [C={channels}, *, F={mel_bins}], got {tuple(lat.shape)}"
                                )
                            truncated.append(lat[:, :quantized_t, :].to(device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                mask = audio_loss_masks_per_item[item_index]
                                if isinstance(mask, torch.Tensor):
                                    mask_out = torch.zeros((quantized_t,), device=device, dtype=torch.float32)
                                    use_mask_t = min(int(mask.shape[0]), quantized_t)
                                    if use_mask_t > 0:
                                        mask_out[:use_mask_t] = mask[:use_mask_t].to(device=device, dtype=torch.float32)
                                    truncated_masks.append(mask_out)
                                else:
                                    truncated_masks.append(torch.ones((quantized_t,), device=device, dtype=torch.float32))
                        else:
                            truncated.append(torch.zeros((channels, quantized_t, mel_bins), device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                truncated_masks.append(torch.zeros((quantized_t,), device=device, dtype=torch.float32))

                    batch_tensor_data["audio_latents"] = torch.stack(truncated)
                    batch_tensor_data["audio_lengths"] = torch.full(
                        (len(truncated),), quantized_t, device=device, dtype=torch.int32
                    )
                    if has_audio_loss_masks:
                        batch_tensor_data["audio_loss_mask"] = torch.stack(truncated_masks)
                else:
                    # Pad mode (default): pad shorter clips to max_t and store actual lengths.
                    lengths = []
                    max_t = 0
                    for i, lat in enumerate(audio_latents_per_item):
                        if isinstance(lat, torch.Tensor):
                            t = int(lat.shape[1])
                            length_val = t
                            cached_len = audio_lengths_per_item[i]
                            if isinstance(cached_len, torch.Tensor) and cached_len.numel() == 1:
                                length_val = int(cached_len.view(-1)[0].item())
                            length_val = max(0, min(length_val, t))
                        else:
                            length_val = 0
                            t = 0
                        lengths.append(length_val)
                        max_t = max(max_t, t)

                    if max_t <= 0:
                        max_t = 1

                    padded = []
                    padded_masks = []
                    has_audio_loss_masks = any(isinstance(mask, torch.Tensor) for mask in audio_loss_masks_per_item)
                    for i, lat in enumerate(audio_latents_per_item):
                        if isinstance(lat, torch.Tensor):
                            if lat.dim() != 3:
                                raise ValueError(f"Expected audio latents to be 3D [C, T, F], got {tuple(lat.shape)}")
                            if int(lat.shape[0]) != channels or int(lat.shape[2]) != mel_bins:
                                raise ValueError(
                                    "Audio latents shape mismatch in batch: "
                                    f"expected [C={channels}, *, F={mel_bins}], got {tuple(lat.shape)}"
                                )

                            t = int(lat.shape[1])
                            use_t = min(t, max_t)
                            out = torch.zeros((channels, max_t, mel_bins), device=device, dtype=dtype)
                            if use_t > 0:
                                out[:, :use_t, :] = lat[:, :use_t, :].to(device=device, dtype=dtype)
                            padded.append(out)
                            lengths[i] = int(min(max(0, lengths[i]), max_t))
                            if has_audio_loss_masks:
                                mask_out = torch.zeros((max_t,), device=device, dtype=torch.float32)
                                mask = audio_loss_masks_per_item[i]
                                if isinstance(mask, torch.Tensor):
                                    use_mask_t = min(int(mask.shape[0]), max_t)
                                    if use_mask_t > 0:
                                        mask_out[:use_mask_t] = mask[:use_mask_t].to(device=device, dtype=torch.float32)
                                else:
                                    valid_t = int(min(max(0, lengths[i]), max_t))
                                    if valid_t > 0:
                                        mask_out[:valid_t] = 1.0
                                padded_masks.append(mask_out)
                        else:
                            padded.append(torch.zeros((channels, max_t, mel_bins), device=device, dtype=dtype))
                            if has_audio_loss_masks:
                                padded_masks.append(torch.zeros((max_t,), device=device, dtype=torch.float32))

                    batch_tensor_data["audio_latents"] = torch.stack(padded)
                    batch_tensor_data["audio_lengths"] = torch.tensor(lengths, device=device, dtype=torch.int32)
                    if has_audio_loss_masks:
                        batch_tensor_data["audio_loss_mask"] = torch.stack(padded_masks)

            else:
                # Skip allocating placeholder audio tensors when the batch has no audio.
                pass

            present_ref_audio = [x for x in ref_audio_latents_per_item if isinstance(x, list) and len(x) > 0]
            if present_ref_audio:
                ref = present_ref_audio[0][0]
                if not isinstance(ref, torch.Tensor) or ref.dim() != 3:
                    raise ValueError(
                        "Expected cached reference audio latents to be 3D [C, T, F] before stacking, "
                        f"got: {getattr(ref, 'shape', None)}"
                    )

                ref_channels = int(ref.shape[0])
                ref_mel_bins = int(ref.shape[2])
                ref_dtype = ref.dtype
                ref_device = ref.device
                max_ref_count = max(len(lat_list) if isinstance(lat_list, list) else 0 for lat_list in ref_audio_latents_per_item)

                if self.audio_bucket_strategy == "truncate":
                    quantized_t = None
                    for elem in reversed(bucket_reso):
                        if isinstance(elem, int):
                            quantized_t = elem
                            break
                    if quantized_t is None or quantized_t <= 0:
                        quantized_t = int(ref.shape[1])

                    truncated_ref = []
                    truncated_ref_lengths = []
                    for lat in ref_audio_latents_per_item:
                        item_refs = []
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            for ref_lat in lat:
                                if ref_lat.dim() != 3:
                                    raise ValueError(f"Expected reference audio latents to be 3D [C, T, F], got {tuple(ref_lat.shape)}")
                                if int(ref_lat.shape[0]) != ref_channels or int(ref_lat.shape[2]) != ref_mel_bins:
                                    raise ValueError(
                                        "Reference audio latents shape mismatch in batch: "
                                        f"expected [C={ref_channels}, *, F={ref_mel_bins}], got {tuple(ref_lat.shape)}"
                                    )
                                item_refs.append(ref_lat[:, :quantized_t, :].to(device=ref_device, dtype=ref_dtype))
                                item_lengths.append(quantized_t)
                        while len(item_refs) < max_ref_count:
                            item_refs.append(torch.zeros((ref_channels, quantized_t, ref_mel_bins), device=ref_device, dtype=ref_dtype))
                            item_lengths.append(0)
                        truncated_ref.append(torch.stack(item_refs))
                        truncated_ref_lengths.append(torch.tensor(item_lengths, device=ref_device, dtype=torch.int32))

                    stacked_ref = torch.stack(truncated_ref)
                    stacked_ref_lengths = torch.stack(truncated_ref_lengths)
                    batch_tensor_data["ref_audio_latents"] = stacked_ref[:, 0] if max_ref_count == 1 else stacked_ref
                    batch_tensor_data["ref_audio_lengths"] = (
                        stacked_ref_lengths[:, 0] if max_ref_count == 1 else stacked_ref_lengths
                    )
                else:
                    ref_lengths = []
                    ref_max_t = 0
                    for i, lat in enumerate(ref_audio_latents_per_item):
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            cached_lengths = ref_audio_lengths_per_item[i] if isinstance(ref_audio_lengths_per_item[i], list) else []
                            for ref_idx, ref_lat in enumerate(lat):
                                t = int(ref_lat.shape[1])
                                length_val = t
                                cached_len = cached_lengths[ref_idx] if ref_idx < len(cached_lengths) else None
                                if isinstance(cached_len, torch.Tensor) and cached_len.numel() == 1:
                                    length_val = int(cached_len.view(-1)[0].item())
                                length_val = max(0, min(length_val, t))
                                item_lengths.append(length_val)
                                ref_max_t = max(ref_max_t, t)
                        ref_lengths.append(item_lengths)

                    if ref_max_t <= 0:
                        ref_max_t = 1

                    padded_ref = []
                    padded_ref_lengths = []
                    for i, lat in enumerate(ref_audio_latents_per_item):
                        item_refs = []
                        item_lengths = []
                        if isinstance(lat, list) and lat:
                            for ref_idx, ref_lat in enumerate(lat):
                                if ref_lat.dim() != 3:
                                    raise ValueError(f"Expected reference audio latents to be 3D [C, T, F], got {tuple(ref_lat.shape)}")
                                if int(ref_lat.shape[0]) != ref_channels or int(ref_lat.shape[2]) != ref_mel_bins:
                                    raise ValueError(
                                        "Reference audio latents shape mismatch in batch: "
                                        f"expected [C={ref_channels}, *, F={ref_mel_bins}], got {tuple(ref_lat.shape)}"
                                    )

                                t = int(ref_lat.shape[1])
                                use_t = min(t, ref_max_t)
                                out = torch.zeros((ref_channels, ref_max_t, ref_mel_bins), device=ref_device, dtype=ref_dtype)
                                if use_t > 0:
                                    out[:, :use_t, :] = ref_lat[:, :use_t, :].to(device=ref_device, dtype=ref_dtype)
                                item_refs.append(out)
                                base_lengths = ref_lengths[i] if i < len(ref_lengths) else []
                                length_val = base_lengths[ref_idx] if ref_idx < len(base_lengths) else 0
                                item_lengths.append(int(min(max(0, length_val), ref_max_t)))
                        while len(item_refs) < max_ref_count:
                            item_refs.append(torch.zeros((ref_channels, ref_max_t, ref_mel_bins), device=ref_device, dtype=ref_dtype))
                            item_lengths.append(0)
                        padded_ref.append(torch.stack(item_refs))
                        padded_ref_lengths.append(torch.tensor(item_lengths, device=ref_device, dtype=torch.int32))

                    stacked_ref = torch.stack(padded_ref)
                    stacked_ref_lengths = torch.stack(padded_ref_lengths)
                    batch_tensor_data["ref_audio_latents"] = stacked_ref[:, 0] if max_ref_count == 1 else stacked_ref
                    batch_tensor_data["ref_audio_lengths"] = (
                        stacked_ref_lengths[:, 0] if max_ref_count == 1 else stacked_ref_lengths
                    )

        if self.timestep_pool is not None:
            batch_tensor_data["timesteps"] = self.timestep_pool[idx][: end - start]  # use the pre-generated timesteps
        else:
            batch_tensor_data["timesteps"] = None

        if batch_count > 0:
            if self.video_loss_weight is not None:
                batch_tensor_data["video_loss_weight"] = float(self.video_loss_weight)
            if self.audio_loss_weight is not None:
                batch_tensor_data["audio_loss_weight"] = float(self.audio_loss_weight)

        if self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            latents = batch_tensor_data.get("latents")
            if isinstance(latents, torch.Tensor) and latents.dim() == 5:
                bsz, _c, frames, height, width = latents.shape
                virtual_num_frames = batch_tensor_data.pop("ltx2_virtual_num_frames", None)
                virtual_height = batch_tensor_data.pop("ltx2_virtual_height", None)
                virtual_width = batch_tensor_data.pop("ltx2_virtual_width", None)

                num_frames_tensor = torch.full((bsz,), frames, dtype=torch.int32)
                height_tensor = torch.full((bsz,), height, dtype=torch.int32)
                width_tensor = torch.full((bsz,), width, dtype=torch.int32)
                if (
                    isinstance(virtual_num_frames, torch.Tensor)
                    and isinstance(virtual_height, torch.Tensor)
                    and isinstance(virtual_width, torch.Tensor)
                ):
                    if (
                        virtual_num_frames.numel() == bsz
                        and virtual_height.numel() == bsz
                        and virtual_width.numel() == bsz
                    ):
                        num_frames_tensor = virtual_num_frames.to(dtype=torch.int32).view(-1)
                        height_tensor = virtual_height.to(dtype=torch.int32).view(-1)
                        width_tensor = virtual_width.to(dtype=torch.int32).view(-1)

                batch_tensor_data["latents"] = {
                    "latents": latents,
                    "num_frames": num_frames_tensor,
                    "height": height_tensor,
                    "width": width_tensor,
                    "fps": torch.full((bsz,), self.target_fps, dtype=torch.float32),
                }

            ref_latents = batch_tensor_data.get("ref_latents")
            if isinstance(ref_latents, torch.Tensor) and ref_latents.dim() == 5:
                bsz, _c, frames, height, width = ref_latents.shape
                batch_tensor_data["ref_latents"] = {
                    "latents": ref_latents,
                    "num_frames": torch.full((bsz,), frames, dtype=torch.int32),
                    "height": torch.full((bsz,), height, dtype=torch.int32),
                    "width": torch.full((bsz,), width, dtype=torch.int32),
                    "fps": torch.full((bsz,), self.target_fps, dtype=torch.float32),
                }

            audio_latents_tensor = batch_tensor_data.get("audio_latents")
            if isinstance(audio_latents_tensor, torch.Tensor) and audio_latents_tensor.dim() == 4:
                batch_tensor_data["audio_latents"] = {"latents": audio_latents_tensor}

            ref_audio_latents_tensor = batch_tensor_data.get("ref_audio_latents")
            if isinstance(ref_audio_latents_tensor, torch.Tensor) and ref_audio_latents_tensor.dim() == 4:
                batch_tensor_data["ref_audio_latents"] = {"latents": ref_audio_latents_tensor}

            video_prompt_embeds = batch_tensor_data.get("video_prompt_embeds")
            audio_prompt_embeds = batch_tensor_data.get("audio_prompt_embeds")
            prompt_attention_mask = batch_tensor_data.get("prompt_attention_mask")

            conditions: dict[str, torch.Tensor] = {}
            if isinstance(video_prompt_embeds, torch.Tensor):
                conditions["video_prompt_embeds"] = video_prompt_embeds
                if isinstance(audio_prompt_embeds, torch.Tensor):
                    conditions["audio_prompt_embeds"] = audio_prompt_embeds
                if isinstance(prompt_attention_mask, torch.Tensor):
                    conditions["prompt_attention_mask"] = prompt_attention_mask

            # Pre-connector features for --train_connectors training
            video_features = batch_tensor_data.get("video_features")
            if isinstance(video_features, torch.Tensor):
                conditions["video_features"] = video_features
            audio_features = batch_tensor_data.get("audio_features")
            if isinstance(audio_features, torch.Tensor):
                conditions["audio_features"] = audio_features

            if not conditions:
                text = batch_tensor_data.get("text")
                text_mask = batch_tensor_data.get("text_mask")
                if isinstance(text, torch.Tensor):
                    if isinstance(text_mask, torch.Tensor):
                        conditions["prompt_attention_mask"] = text_mask

                    # Legacy cache fallback: keep full prompt_embeds so runtime can
                    # split by model dims (video/audio) when needed.
                    conditions["prompt_embeds"] = text
                    conditions["video_prompt_embeds"] = text

            # DINOv2 features (pre-cached, for CREPA dino mode)
            if any(d is not None for d in dino_features_per_item):
                if all(d is not None for d in dino_features_per_item):
                    # Pad to max T_pixel and stack
                    # Features are [T, N_patches, D] (patch tokens)
                    max_t = max(d.shape[0] for d in dino_features_per_item)
                    padded = []
                    for d in dino_features_per_item:
                        if d.shape[0] < max_t:
                            # Pad along T dim only; N_patches and D are constant
                            pad_shape = [max_t - d.shape[0]] + list(d.shape[1:])
                            pad = torch.zeros(pad_shape, dtype=d.dtype)
                            d = torch.cat([d, pad], dim=0)
                        padded.append(d)
                    conditions["dino_features"] = torch.stack(padded)  # [B, T_pixel, N_patches, D_dino]

            if conditions:
                batch_tensor_data["conditions"] = conditions

        if diag_collect_keys:
            batch_tensor_data["item_keys"] = item_keys
            batch_tensor_data["latent_cache_paths"] = latent_cache_paths
            batch_tensor_data["audio_cache_paths"] = audio_cache_paths
            batch_tensor_data["text_cache_paths"] = text_cache_paths
        batch_tensor_data["captions"] = captions

        return batch_tensor_data


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, list[Image.Image]]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


class AudioDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_audio_data(self, idx: int) -> tuple[str, str]:
        """
        Returns audio data as a tuple of audio path and caption.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: Optional[int] = None,
        multiple_target: bool = False,
        loss_mask_directory: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.loss_mask_directory = loss_mask_directory
        self.loss_mask_use_alpha = loss_mask_use_alpha
        self.loss_mask_invert = loss_mask_invert
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory, caption_extension=self.caption_extension)
        logger.info(f"found {len(self.image_paths)} images")

        # check if multiple-target images exist
        self.target_paths: dict[str, list[str]] = {}  # image_path -> list of target image paths

        if self.multiple_target:
            # sort by length, longer first
            sorted_image_paths = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            all_image_paths = set(glob_images(self.image_directory))  # image1.jpg, image1_1.jpg, image1_2.jpg, ...
            multiple_target_candidates = all_image_paths - set(sorted_image_paths)  # those not in the images with captions

            if len(multiple_target_candidates) > 0:
                logger.info("checking for multiple-target images")
                for image_path in sorted_image_paths:
                    image_path_no_ext = os.path.splitext(image_path)[0]

                    # find matching multiple-target images
                    potential_paths = [p for p in multiple_target_candidates if p.startswith(image_path_no_ext + "_")]

                    if potential_paths:
                        # sort by the digits (`_0000`) suffix
                        def sort_key(path):
                            path_no_ext = os.path.splitext(path)[0]
                            digits_suffix = path_no_ext.rsplit("_", 1)[-1]
                            if not digits_suffix.isdigit():
                                raise ValueError(
                                    f"Invalid digits suffix in '{path_no_ext}'. Expected a numeric suffix after '_' "
                                    f"(e.g., '_0', '_1', '_2') for proper sorting of multiple target images."
                                )
                            return int(digits_suffix)

                        potential_paths.sort(key=sort_key)
                        self.target_paths[image_path] = potential_paths

                        # remove to avoid duplicate matching
                        multiple_target_candidates.difference_update(potential_paths)

                # check the number of targets: all multiple-target images should have the same number of targets
                num_targets = 0
                for image_path, paths in self.target_paths.items():
                    if num_targets == 0:
                        num_targets = len(paths)
                    elif num_targets != len(paths):
                        logger.error(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )
                        raise ValueError(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )

                if num_targets == 0:
                    logger.error("no multiple-target images found, but multiple_target is set to True")
                    raise ValueError("no multiple-target images found, but multiple_target is set to True")

                logger.info(f"found multiple-target images, max targets per image: {num_targets}")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}

            # sort image paths for matching control images properly: longer names first
            image_paths_sorted = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            # glob control images first
            all_control_image_paths = set(glob_images(self.control_directory))

            for image_path in image_paths_sorted:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]

                # find matching control images
                potential_paths = [
                    p
                    for p in all_control_image_paths
                    if os.path.basename(p).startswith(image_basename_no_ext + ".")
                    or os.path.basename(p).startswith(image_basename_no_ext + "_")
                ]

                # remove to avoid duplicate matching
                all_control_image_paths.difference_update(potential_paths)

                if potential_paths:
                    # sort by the digits (`_0000`) suffix, prefer the one without the suffix
                    def sort_key(path):
                        basename = os.path.basename(path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        if image_basename_no_ext == basename_no_ext:  # prefer the one without suffix
                            return 0
                        digits_suffix = basename_no_ext.rsplit("_", 1)[-1]
                        if not digits_suffix.isdigit():
                            raise ValueError(f"Invalid digits suffix in {basename_no_ext}")
                        return int(digits_suffix) + 1

                    potential_paths.sort(key=sort_key)
                    if control_count_per_image is not None and len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = (
                        potential_paths[:control_count_per_image] if control_count_per_image is not None else potential_paths
                    )
            logger.info(
                f"found {len(self.control_paths)} matching control images for {'arbitrary' if control_count_per_image is None else control_count_per_image} images"
            )

            # log the distribution of number of control images
            count_of_num_control_images = {}
            for paths in self.control_paths.values():
                count = len(paths)
                if count not in count_of_num_control_images:
                    count_of_num_control_images[count] = 0
                count_of_num_control_images[count] += 1
            for count, num_images in count_of_num_control_images.items():
                logger.info(f"  {num_images} images have {count} control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]], Optional[Image.Image]]:
        image_path = self.image_paths[idx]
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            image_paths += self.target_paths.get(image_path, [])

        images = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        _, caption = self.get_caption(idx)

        loss_mask = None
        if self.loss_mask_directory is not None:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            loss_mask_path = find_stem_matched_file(self.loss_mask_directory, stem, IMAGE_EXTENSIONS)
            if loss_mask_path is not None:
                loss_mask = load_loss_mask_image(loss_mask_path, invert=self.loss_mask_invert)
        elif self.loss_mask_use_alpha:
            loss_mask = alpha_channel_to_loss_mask(images[0], invert=self.loss_mask_invert)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls, loss_mask

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_image_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_image_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class ImageJsonlDatasource(ImageDatasource):
    def __init__(
        self,
        image_jsonl_file: str,
        control_count_per_image: Optional[int] = None,
        multiple_target: bool = False,
        caption_field: Optional[str] = None,
        loss_mask_directory: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
    ):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.caption_field = caption_field
        self.loss_mask_directory = loss_mask_directory
        self.loss_mask_use_alpha = loss_mask_use_alpha
        self.loss_mask_invert = loss_mask_invert
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.image_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            if self.control_count_per_image is None:
                logger.info(f"found {len(self.data)} images with arbitrary control images per image in JSONL data")
            else:
                missing_control_images = [
                    item["image_path"]
                    for item in self.data
                    if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                ]
                if missing_control_images:
                    logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                    raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                logger.info(
                    f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data"
                )

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]], Optional[Image.Image]]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            while True:
                next_index = len(image_paths)  # start from 1
                next_image_path = data.get("image_path_" + str(next_index), None)
                if next_image_path is None:
                    break
                if not os.path.exists(next_image_path):
                    raise ValueError(f"multiple-target image not found: {next_image_path}")

                image_paths.append(next_image_path)

        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        caption = select_caption_from_metadata(data, self.caption_field)

        loss_mask = None
        mask_path = data.get("loss_mask_path") or data.get("image_loss_mask_path")
        if mask_path:
            loss_mask = load_loss_mask_image(mask_path, invert=self.loss_mask_invert)
        elif self.loss_mask_directory is not None:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            loss_mask_path = find_stem_matched_file(self.loss_mask_directory, stem, IMAGE_EXTENSIONS)
            if loss_mask_path is not None:
                loss_mask = load_loss_mask_image(loss_mask_path, invert=self.loss_mask_invert)
        elif self.loss_mask_use_alpha:
            loss_mask = alpha_channel_to_loss_mask(images[0], invert=self.loss_mask_invert)

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image or 1000):  # arbitrary large number if control_count_per_image is None
                if f"control_path_{i}" not in data:
                    break
                control_path = data[f"control_path_{i}"]
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls, loss_mask

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        caption = select_caption_from_metadata(data, self.caption_field)
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class AudioDirectoryDatasource(AudioDatasource):
    def __init__(
        self,
        audio_directory: str,
        caption_extension: Optional[str] = None,
        loss_mask_directory: Optional[str] = None,
    ):
        super().__init__()
        self.audio_directory = audio_directory
        self.caption_extension = caption_extension
        self.loss_mask_directory = loss_mask_directory
        self.current_idx = 0

        logger.info(f"glob audio in {self.audio_directory}")
        self.audio_paths = glob_audio(self.audio_directory)
        logger.info(f"found {len(self.audio_paths)} audio files")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.audio_paths)

    def get_audio_data(self, idx: int) -> tuple[str, str, Optional[list[tuple[float, float]]]]:
        audio_path = self.audio_paths[idx]
        caption_path = os.path.splitext(audio_path)[0] + (self.caption_extension or "")
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        intervals = None
        if self.loss_mask_directory is not None:
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            mask_path = find_stem_matched_file(self.loss_mask_directory, stem, MASK_METADATA_EXTENSIONS)
            if mask_path is not None and os.path.isfile(mask_path):
                intervals = load_audio_loss_mask_intervals(mask_path)
        return audio_path, caption, intervals

    def get_caption(self, idx: int) -> tuple[str, str]:
        audio_path, caption, _intervals = self.get_audio_data(idx)
        return audio_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.audio_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_audio_fetcher(index):
                return lambda: self.get_audio_data(index)

            fetcher = create_audio_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class AudioJsonlDatasource(AudioDatasource):
    def __init__(
        self,
        audio_jsonl_file: str,
        caption_field: Optional[str] = None,
        loss_mask_directory: Optional[str] = None,
    ):
        super().__init__()
        self.audio_jsonl_file = audio_jsonl_file
        self.caption_field = caption_field
        self.loss_mask_directory = loss_mask_directory
        self.current_idx = 0

        logger.info(f"load audio jsonl from {self.audio_jsonl_file}")
        self.data = []
        with open(self.audio_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.audio_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} audio items")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_audio_data(self, idx: int) -> tuple[str, str, Optional[list[tuple[float, float]]]]:
        data = self.data[idx]
        audio_path = data["audio_path"]
        caption = select_caption_from_metadata(data, self.caption_field)
        intervals = normalize_loss_mask_intervals(data.get("loss_mask_intervals") or data.get("audio_loss_mask_intervals"))
        mask_path = data.get("loss_mask_path") or data.get("audio_loss_mask_path")
        if mask_path:
            intervals = load_audio_loss_mask_intervals(mask_path)
        elif intervals is None and self.loss_mask_directory is not None:
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            mask_path = find_stem_matched_file(self.loss_mask_directory, stem, MASK_METADATA_EXTENSIONS)
            if mask_path is not None and os.path.isfile(mask_path):
                intervals = load_audio_loss_mask_intervals(mask_path)
        return audio_path, caption, intervals

    def get_caption(self, idx: int) -> tuple[str, str]:
        audio_path, caption, _intervals = self.get_audio_data(idx)
        return audio_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_audio_fetcher(index):
                return lambda: self.get_audio_data(index)

            fetcher = create_audio_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        video = load_video(
            video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return video

    def get_control_data_from_path(
        self,
        control_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        control = load_video(
            control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return control

    def set_start_and_end_frame(self, start_frame: Optional[int], end_frame: Optional[int]):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(self, source_fps: Optional[float], target_fps: Optional[float]):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
    def __init__(
        self,
        video_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        loss_mask_directory: Optional[str] = None,
        loss_mask_invert: bool = False,
    ):
        super().__init__()
        self.video_directory = video_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory  # 新しく追加: コントロール画像ディレクトリ
        self.loss_mask_directory = loss_mask_directory
        self.loss_mask_invert = loss_mask_invert
        self.current_idx = 0

        # glob videos
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control videos in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for video_path in self.video_paths:
                video_basename = os.path.basename(video_path)
                # construct control path from video path
                # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mp4"
                control_path = os.path.join(self.control_directory, video_basename)
                if os.path.exists(control_path):
                    self.control_paths[video_path] = control_path
                else:
                    # use the same base name for control path
                    base_name = os.path.splitext(video_basename)[0]

                    # directory with images. for example: video_path = "vid/video.mp4" -> control_path = "control/video"
                    potential_path = os.path.join(self.control_directory, base_name)  # no extension
                    if os.path.isdir(potential_path):
                        self.control_paths[video_path] = potential_path
                    else:
                        # another extension for control path
                        # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mov"
                        for ext in VIDEO_EXTENSIONS:
                            potential_path = os.path.join(self.control_directory, base_name + ext)
                            if os.path.exists(potential_path):
                                self.control_paths[video_path] = potential_path
                                break

            logger.info(f"found {len(self.control_paths)} matching control videos/images")
            # check if all videos have matching control paths, if not, raise an error
            missing_controls = len(self.video_paths) - len(self.control_paths)
            if missing_controls > 0:
                # logger.warning(f"Could not find matching control videos/images for {missing_controls} videos")
                missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
                logger.error(
                    f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}"
                )
                raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]], Optional[list[np.ndarray]]]:
        video_path = self.video_paths[idx]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        _, caption = self.get_caption(idx)

        control = None
        if self.control_directory is not None and video_path in self.control_paths:
            control_path = self.control_paths[video_path]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        loss_mask = None
        if self.loss_mask_directory is not None and video:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            mask_path = find_stem_matched_file(self.loss_mask_directory, stem)
            if mask_path is not None:
                bucket_reso = (video[0].shape[1], video[0].shape[0])
                loss_mask = load_loss_mask_frames(
                    mask_path,
                    bucket_reso=bucket_reso,
                    frame_count=len(video),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    source_fps=self.source_fps,
                    target_fps=self.target_fps,
                    invert=self.loss_mask_invert,
                )

        return video_path, video, caption, control, loss_mask

    def get_caption(self, idx: int) -> tuple[str, str]:
        video_path = self.video_paths[idx]
        caption_path = os.path.splitext(video_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoJsonlDatasource(VideoDatasource):
    def __init__(
        self,
        video_jsonl_file: str,
        caption_field: Optional[str] = None,
        loss_mask_directory: Optional[str] = None,
        loss_mask_invert: bool = False,
    ):
        super().__init__()
        self.video_jsonl_file = video_jsonl_file
        self.caption_field = caption_field
        self.loss_mask_directory = loss_mask_directory
        self.loss_mask_invert = loss_mask_invert
        self.current_idx = 0

        # load jsonl
        logger.info(f"load video jsonl from {self.video_jsonl_file}")
        self.data = []
        with open(self.video_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} videos")

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path" in item for item in self.data)
        if self.has_control:
            control_count = sum(1 for item in self.data if "control_path" in item)
            if control_count < len(self.data):
                missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
                logger.error(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
                raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
            logger.info(f"found {control_count} control videos/images in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]], Optional[list[np.ndarray]]]:
        data = self.data[idx]
        video_path = data["video_path"]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        caption = select_caption_from_metadata(data, self.caption_field)

        control = None
        if "control_path" in data and data["control_path"]:
            control_path = data["control_path"]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        loss_mask = None
        mask_path = data.get("loss_mask_path") or data.get("video_loss_mask_path")
        if not mask_path and self.loss_mask_directory is not None:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            mask_path = find_stem_matched_file(self.loss_mask_directory, stem)
        if mask_path and video:
            bucket_reso = (video[0].shape[1], video[0].shape[0])
            loss_mask = load_loss_mask_frames(
                mask_path,
                bucket_reso=bucket_reso,
                frame_count=len(video),
                start_frame=start_frame,
                end_frame=end_frame,
                source_fps=self.source_fps,
                target_fps=self.target_fps,
                invert=self.loss_mask_invert,
            )

        return video_path, video, caption, control, loss_mask

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        video_path = data["video_path"]
        caption = select_caption_from_metadata(data, self.caption_field)
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        caption_field: Optional[str] = None,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
        cache_directory: Optional[str] = None,
        reference_cache_directory: Optional[str] = None,
        reference_cache_directories: Optional[Sequence[str]] = None,
        reference_audio_cache_directory: Optional[str] = None,
        reference_audio_cache_directories: Optional[Sequence[str]] = None,
        separate_audio_buckets: bool = False,
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.caption_field = caption_field
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.cache_directory = cache_directory
        self.reference_cache_directories = _normalize_optional_path_list(
            reference_cache_directory,
            reference_cache_directories,
        )
        self.reference_cache_directory = self.reference_cache_directories[0] if self.reference_cache_directories else None
        self.reference_audio_cache_directories = _normalize_optional_path_list(
            reference_audio_cache_directory,
            reference_audio_cache_directories,
        )
        self.reference_audio_cache_directory = (
            self.reference_audio_cache_directories[0] if self.reference_audio_cache_directories else None
        )
        self.separate_audio_buckets = separate_audio_buckets
        self.loss_mask_directory = loss_mask_directory
        self.default_loss_mask_path = default_loss_mask_path
        self.loss_mask_use_alpha = loss_mask_use_alpha
        self.loss_mask_invert = loss_mask_invert
        self.debug_dataset = debug_dataset
        self.architecture = architecture
        self.reference_downscale = 1
        self.seed = None
        self.current_epoch = 0
        self.shared_epoch = None

        if not self.enable_bucket:
            self.bucket_no_upscale = False

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "caption_field": self.caption_field,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
            "separate_audio_buckets": bool(self.separate_audio_buckets),
        }
        return metadata

    def get_audio_latent_cache_path_from_latent_cache_path(self, latent_cache_path: str) -> str:
        base_dir = os.path.dirname(latent_cache_path)
        base_name = os.path.basename(latent_cache_path)
        suffix = f"_{self.architecture}.safetensors"
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)] + f"_{self.architecture}_audio.safetensors"
            return os.path.join(base_dir, base_name)
        stem, _ext = os.path.splitext(base_name)
        return os.path.join(base_dir, f"{stem}_{self.architecture}_audio.safetensors")

    def get_audio_latent_cache_path(self, item_info: ItemInfo) -> str:
        latent_cache_path = getattr(item_info, "latent_cache_path", None)
        if not latent_cache_path:
            latent_cache_path = self.get_latent_cache_path(item_info)
        return self.get_audio_latent_cache_path_from_latent_cache_path(latent_cache_path)

    def get_dino_feature_cache_path_from_latent_cache_path(self, latent_cache_path: str) -> str:
        """Derive DINOv2 feature cache path: ``*_ltx2.safetensors`` → ``*_ltx2_dino.safetensors``."""
        base_dir = os.path.dirname(latent_cache_path)
        base_name = os.path.basename(latent_cache_path)
        suffix = f"_{self.architecture}.safetensors"
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)] + f"_{self.architecture}_dino.safetensors"
            return os.path.join(base_dir, base_name)
        stem, _ext = os.path.splitext(base_name)
        return os.path.join(base_dir, f"{stem}_{self.architecture}_dino.safetensors")

    def _append_audio_bucket_key(self, bucket_key: tuple[Any, ...], has_audio: bool) -> tuple[Any, ...]:
        if not self.separate_audio_buckets:
            return bucket_key
        if self.architecture not in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            return bucket_key
        return (*bucket_key, bool(has_audio))

    def get_all_latent_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

    def get_all_text_encoder_output_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_te.safetensors"))

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        """
        Returns the cache path for the latent tensor.

        item_info: ItemInfo object

        Returns:
            str: cache path

        cache_path is based on the item_key and the resolution.
        """
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")

    def get_reference_latent_cache_path(self, item_info: ItemInfo) -> str:
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.reference_cache_directory is not None, (
            "reference_cache_directory is required / reference_cache_directoryは必須です"
        )
        return os.path.join(
            self.reference_cache_directory,
            f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors",
        )

    def get_reference_latent_cache_paths(self, item_info: ItemInfo) -> list[str]:
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.reference_cache_directories, (
            "reference_cache_directories is required / reference_cache_directoriesは必須です"
        )
        return [
            os.path.join(directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")
            for directory in self.reference_cache_directories
        ]

    def get_reference_audio_latent_cache_path(self, item_info: ItemInfo) -> str:
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.reference_audio_cache_directory is not None, (
            "reference_audio_cache_directory is required / reference_audio_cache_directoryは必須です"
        )
        return os.path.join(
            self.reference_audio_cache_directory,
            f"{basename}_{w:04d}x{h:04d}_{self.architecture}_audio.safetensors",
        )

    def get_reference_audio_latent_cache_paths(self, item_info: ItemInfo) -> list[str]:
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.reference_audio_cache_directories, (
            "reference_audio_cache_directories is required / reference_audio_cache_directoriesは必須です"
        )
        return [
            os.path.join(directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}_audio.safetensors")
            for directory in self.reference_audio_cache_directories
        ]

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{self.architecture}_te.safetensors")

    def retrieve_latent_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        pass

    def set_seed(self, seed: int, shared_epoch: SharedEpoch):
        self.seed = seed
        self.shared_epoch = shared_epoch

    def set_current_epoch(self, epoch):
        assert self.shared_epoch is not None, "shared_epoch is None"
        assert self.shared_epoch.value == epoch, "shared_epoch does not match"

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        assert self.shared_epoch is not None, "shared_epoch is None"
        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            logger.info(f"epoch is incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            num_epochs = epoch - self.current_epoch
            for _ in range(num_epochs):
                self.current_epoch += 1
                self.shuffle_buckets()
        elif epoch < self.current_epoch:
            logger.warning(f"epoch is not incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            self.current_epoch = epoch

    def _default_retrieve_text_encoder_output_cache_batches(self, datasource: ContentDatasource, batch_size: int, num_workers: int):
        datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    data.append(item_info)

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= batch_size or (len(data) > 0 and flush):
                batch = data[0:batch_size]
                if len(data) > batch_size:
                    data = data[batch_size:]
                else:
                    data = []
                return batch
            return None

        for fetch_op in datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()


class ImageDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
        caption_field: Optional[str] = None,
        image_directory: Optional[str] = None,
        image_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        multiple_target: bool = False,
        reference_cache_directory: Optional[str] = None,
        reference_cache_directories: Optional[Sequence[str]] = None,
        reference_audio_cache_directory: Optional[str] = None,
        reference_audio_cache_directories: Optional[Sequence[str]] = None,
        separate_audio_buckets: bool = False,
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
        fp_latent_window_size: Optional[int] = 9,
        fp_1f_clean_indices: Optional[list[int]] = None,
        fp_1f_target_index: Optional[int] = None,
        fp_1f_no_post: Optional[bool] = False,
        no_resize_control: Optional[bool] = False,
        control_resolution: Optional[Tuple[int, int]] = None,
        cache_only: bool = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        super(ImageDataset, self).__init__(
            resolution,
            caption_extension,
            caption_field,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            video_loss_weight,
            audio_loss_weight,
            cache_directory,
            reference_cache_directory,
            reference_cache_directories,
            reference_audio_cache_directory,
            reference_audio_cache_directories,
            separate_audio_buckets,
            loss_mask_directory,
            default_loss_mask_path,
            loss_mask_use_alpha,
            loss_mask_invert,
            debug_dataset,
            architecture,
        )
        self.image_directory = image_directory
        self.image_jsonl_file = image_jsonl_file
        self.control_directory = control_directory
        self.multiple_target = multiple_target
        self.fp_latent_window_size = fp_latent_window_size
        self.fp_1f_clean_indices = fp_1f_clean_indices
        self.fp_1f_target_index = fp_1f_target_index
        self.fp_1f_no_post = fp_1f_no_post
        self.no_resize_control = no_resize_control
        self.control_resolution = control_resolution
        self.cache_only = cache_only

        control_count_per_image: Optional[int] = 1
        if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
            if fp_1f_clean_indices is not None:
                control_count_per_image = len(fp_1f_clean_indices)
            else:
                control_count_per_image = 1
        elif self.architecture == ARCHITECTURE_FLUX_KONTEXT:
            control_count_per_image = 1
        elif (
            self.architecture == ARCHITECTURE_FLUX_2_DEV
            or self.architecture == ARCHITECTURE_FLUX_2_KLEIN_4B
            or self.architecture == ARCHITECTURE_FLUX_2_KLEIN_9B
        ):
            control_count_per_image = None  # can be multiple control images
        elif self.architecture == ARCHITECTURE_QWEN_IMAGE_EDIT:
            control_count_per_image = None  # can be multiple control images

        if self.cache_only:
            self.datasource = None
        elif image_directory is not None:
            self.datasource = ImageDirectoryDatasource(
                image_directory,
                caption_extension,
                control_directory,
                control_count_per_image,
                multiple_target,
                loss_mask_directory=loss_mask_directory,
                loss_mask_use_alpha=loss_mask_use_alpha,
                loss_mask_invert=loss_mask_invert,
            )
        elif image_jsonl_file is not None:
            self.datasource = ImageJsonlDatasource(
                image_jsonl_file,
                control_count_per_image,
                multiple_target,
                caption_field=caption_field,
                loss_mask_directory=loss_mask_directory,
                loss_mask_use_alpha=loss_mask_use_alpha,
                loss_mask_invert=loss_mask_invert,
            )
        else:
            raise ValueError("image_directory or image_jsonl_file must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.image_directory
        if self.cache_only and self.cache_directory is None:
            raise ValueError("cache_directory is required when cache_only=True")

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control if self.datasource is not None else False

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        if self.image_jsonl_file is not None:
            metadata["image_jsonl_file"] = os.path.basename(self.image_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["has_control"] = self.has_control
        metadata["cache_only"] = self.cache_only
        return metadata

    def get_total_image_count(self):
        if self.datasource is None:
            return None
        return len(self.datasource) if self.datasource.is_indexable() else None

    def retrieve_latent_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_latent_cache_batches is not available when cache_only=True")
        bucket_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
            self.architecture,
            reference_downscale=getattr(self, "reference_downscale", 1),
        )
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[tuple[int, int], list[ItemInfo]] = {}  # (width, height) -> [ItemInfo]
        futures = []

        # aggregate futures and sort by bucket resolution
        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_size, item_key, images, caption, controls, loss_mask = future.result()
                    image = images[0]  # use the first image as the main content
                    bucket_height, bucket_width = image.shape[:2]
                    bucket_reso = (bucket_width, bucket_height)

                    item_info = ItemInfo(
                        item_key, caption, original_size, bucket_reso, content=image if len(images) == 1 else images
                    )
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)

                    if self.reference_cache_directories:
                        item_info.reference_latent_cache_paths = self.get_reference_latent_cache_paths(item_info)
                        item_info.reference_latent_cache_path = item_info.reference_latent_cache_paths[0]

                    # for VLM, which require image in addition to text, like Qwen-Image-Edit
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)

                    item_info.fp_latent_window_size = self.fp_latent_window_size
                    item_info.fp_1f_clean_indices = self.fp_1f_clean_indices
                    item_info.fp_1f_target_index = self.fp_1f_target_index
                    item_info.fp_1f_no_post = self.fp_1f_no_post

                    if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                        # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                        bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                        if self.fp_1f_clean_indices is not None:
                            bucket_reso.append(len(self.fp_1f_clean_indices))
                            bucket_reso.append(self.fp_1f_no_post)
                        bucket_reso = tuple(bucket_reso)

                    if controls is not None:
                        item_info.control_content = controls
                        if self.no_resize_control or self.control_resolution is not None:
                            # Add control size to bucket_reso to make different control resolutions to different batch
                            bucket_reso = list(bucket_reso)
                            for control in controls:
                                bucket_reso = bucket_reso + list(control.shape[0:2])
                            bucket_reso = tuple(bucket_reso)

                    if loss_mask is not None:
                        item_info.loss_mask_content = loss_mask

                    if bucket_reso not in batches:
                        batches[bucket_reso] = []
                    batches[bucket_reso].append(item_info)

                    futures.remove(future)

        # submit batch if some bucket has enough items
        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:
            # fetch and resize image in a separate thread
            def fetch_and_resize(op: callable) -> tuple[tuple[int, int], str, list[np.ndarray], str, Optional[list[np.ndarray]], Optional[np.ndarray]]:
                result = op()
                if len(result) == 4:
                    image_key, images, caption, controls = result
                    loss_mask = None
                else:
                    image_key, images, caption, controls, loss_mask = result
                images: list[Image.Image]
                image: Image.Image = images[0]  # use the first image as the main content
                image_size = image.size

                bucket_reso = bucket_selector.get_bucket_resolution(image_size)
                images = [resize_image_to_bucket(img, bucket_reso) for img in images]  # list of np.ndarray

                resized_loss_mask = None
                if loss_mask is not None:
                    resized_loss_mask = loss_mask_to_float_array(loss_mask, bucket_reso)
                elif self.default_loss_mask_path:
                    resized_loss_mask = loss_mask_to_float_array(
                        load_loss_mask_image(self.default_loss_mask_path, invert=self.loss_mask_invert),
                        bucket_reso,
                    )

                resized_controls = None
                if controls is not None:
                    resized_controls = []
                    if self.no_resize_control:
                        for control in controls:
                            # divisible by bucket reso steps
                            width, height = control.size

                            if self.control_resolution is not None:
                                # use control resolution as maximum
                                max_width, max_height = self.control_resolution
                                if width * height > max_width * max_height:
                                    width, height = BucketSelector.calculate_bucket_resolution(
                                        control.size,
                                        self.control_resolution,
                                        architecture=self.architecture,
                                        reference_downscale=getattr(self, "reference_downscale", 1),
                                    )
                            else:
                                width = width - (width % bucket_selector.reso_steps)
                                height = height - (height % bucket_selector.reso_steps)

                            resized_control = resize_image_to_bucket(control, (width, height))  # returns np.ndarray
                            resized_controls.append(resized_control)
                    elif self.control_resolution is not None:
                        for control in controls:
                            control_bucket_reso = BucketSelector.calculate_bucket_resolution(
                                control.size,
                                self.control_resolution,
                                architecture=self.architecture,
                                reference_downscale=getattr(self, "reference_downscale", 1),
                            )
                            resized_control = resize_image_to_bucket(control, control_bucket_reso)
                            resized_controls.append(resized_control)
                    else:
                        for control in controls:
                            resized_control = resize_image_to_bucket(control, bucket_reso)
                            resized_controls.append(resized_control)

                return image_size, image_key, images, caption, resized_controls, resized_loss_mask

            future = executor.submit(fetch_and_resize, fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_text_encoder_output_cache_batches is not available when cache_only=True")
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        bucket_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
            self.architecture,
            reference_downscale=getattr(self, "reference_downscale", 1),
        )

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        # (width, height) -> [ItemInfo] or (width, height, other conds...) -> [ItemInfo]
        bucketed_item_info: dict[Union[tuple[int, int], Any], list[ItemInfo]] = {}
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            audio_latent_cache_file = self.get_audio_latent_cache_path_from_latent_cache_path(cache_file)

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)

            if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                if self.fp_1f_clean_indices is not None:
                    bucket_reso.append(len(self.fp_1f_clean_indices))
                    bucket_reso.append(self.fp_1f_no_post)
                bucket_reso = tuple(bucket_reso)
            if self.no_resize_control or self.control_resolution is not None:
                # we also need to split the bucket with control resolutions
                control_key = safetensors_utils.find_key(cache_file, starts_with="latents_control_")  # latents_control_FxHxW_dtype
                if control_key is not None:
                    control_shape = control_key.rsplit("_", 3)[-2]  # FxHxW
                    bucket_reso = tuple(list(bucket_reso) + [control_shape])  # (int, int, str)

            has_audio = os.path.exists(audio_latent_cache_file)
            bucket_reso = self._append_audio_bucket_key(tuple(bucket_reso), has_audio)
            item_info = ItemInfo(item_key, "", image_size, bucket_reso, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file
            item_info.audio_latent_cache_path = audio_latent_cache_file if has_audio else None

            dino_cache_file = self.get_dino_feature_cache_path_from_latent_cache_path(cache_file)
            item_info.dino_feature_cache_path = dino_cache_file if os.path.exists(dino_cache_file) else None

            if self.reference_cache_directories:
                reference_cache_paths: list[str] = []
                missing_reference_cache = False
                for reference_cache_directory in self.reference_cache_directories:
                    ref_cache_path = os.path.join(reference_cache_directory, os.path.basename(cache_file))
                    if os.path.exists(ref_cache_path):
                        reference_cache_paths.append(ref_cache_path)
                    else:
                        logger.warning(f"Reference cache not found, skipping item: {ref_cache_path}")
                        missing_reference_cache = True
                        break
                if missing_reference_cache:
                    continue
                if reference_cache_paths:
                    item_info.reference_latent_cache_paths = reference_cache_paths
                    item_info.reference_latent_cache_path = reference_cache_paths[0]

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(
            bucketed_item_info,
            self.batch_size,
            num_timestep_buckets=num_timestep_buckets,
            architecture=self.architecture,
            video_loss_weight=self.video_loss_weight,
            audio_loss_weight=self.audio_loss_weight,
        )
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


class AudioDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
        caption_field: Optional[str] = None,
        audio_directory: Optional[str] = None,
        audio_jsonl_file: Optional[str] = None,
        cache_directory: Optional[str] = None,
        reference_cache_directory: Optional[str] = None,
        reference_cache_directories: Optional[Sequence[str]] = None,
        reference_audio_cache_directory: Optional[str] = None,
        reference_audio_cache_directories: Optional[Sequence[str]] = None,
        separate_audio_buckets: bool = False,
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
        cache_only: bool = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        audio_bucket_strategy: str = "pad",
        audio_bucket_interval: float = 2.0,
    ):
        super(AudioDataset, self).__init__(
            resolution,
            caption_extension,
            caption_field,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            video_loss_weight,
            audio_loss_weight,
            cache_directory,
            reference_cache_directory,
            reference_cache_directories,
            reference_audio_cache_directory,
            reference_audio_cache_directories,
            separate_audio_buckets,
            loss_mask_directory,
            default_loss_mask_path,
            loss_mask_use_alpha,
            loss_mask_invert,
            debug_dataset,
            architecture,
        )
        self.audio_directory = audio_directory
        self.audio_jsonl_file = audio_jsonl_file
        self.cache_only = cache_only
        self.audio_bucket_strategy = audio_bucket_strategy
        self.audio_bucket_interval = audio_bucket_interval

        if self.audio_bucket_strategy not in ("pad", "truncate"):
            raise ValueError(f"audio_bucket_strategy must be 'pad' or 'truncate', got '{self.audio_bucket_strategy}'")

        if self.cache_only:
            self.datasource = None
        elif audio_directory is not None:
            self.datasource = AudioDirectoryDatasource(audio_directory, caption_extension, loss_mask_directory=loss_mask_directory)
        elif audio_jsonl_file is not None:
            self.datasource = AudioJsonlDatasource(
                audio_jsonl_file,
                caption_field=caption_field,
                loss_mask_directory=loss_mask_directory,
            )
        else:
            raise ValueError("audio_directory or audio_jsonl_file must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.audio_directory
        if self.cache_only and self.cache_directory is None:
            raise ValueError("cache_directory is required when cache_only=True")

        self.batch_manager = None
        self.num_train_items = 0

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.audio_directory is not None:
            metadata["audio_directory"] = os.path.basename(self.audio_directory)
        if self.audio_jsonl_file is not None:
            metadata["audio_jsonl_file"] = os.path.basename(self.audio_jsonl_file)
        metadata["cache_only"] = self.cache_only
        return metadata

    def _uses_ltx2_audio_video_geometry(self) -> bool:
        return self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}

    def _legacy_audio_latent_cache_path(self, item_key: str) -> str:
        basename = os.path.splitext(os.path.basename(item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_0001x0001_{self.architecture}.safetensors")

    def _legacy_strip_resolution_suffix(self, item_key: str) -> str:
        suffix = "_0001x0001"
        return item_key[: -len(suffix)] if item_key.endswith(suffix) else item_key

    def retrieve_latent_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_latent_cache_batches is not available when cache_only=True")
        executor = ThreadPoolExecutor(max_workers=num_workers)
        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:
                        time.sleep(0.1)
                        continue
                    break

                for future in completed_futures:
                    result = future.result()
                    if len(result) == 2:
                        audio_path, caption = result
                        loss_mask_intervals = None
                    else:
                        audio_path, caption, loss_mask_intervals = result
                    if self._uses_ltx2_audio_video_geometry():
                        width, height = int(self.resolution[0]), int(self.resolution[1])
                        bucket_reso = self._append_audio_bucket_key((width, height), True)
                        item_info = ItemInfo(audio_path, caption, (width, height), bucket_reso)
                        item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                    else:
                        bucket_reso = self._append_audio_bucket_key((1, 1), True)
                        item_info = ItemInfo(audio_path, caption, (1, 1), bucket_reso)
                        item_info.latent_cache_path = self._legacy_audio_latent_cache_path(audio_path)
                    item_info.audio_latent_cache_path = self.get_audio_latent_cache_path(item_info)
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    item_info.audio_path = audio_path
                    if loss_mask_intervals is None and self.default_loss_mask_path:
                        loss_mask_intervals = load_audio_loss_mask_intervals(self.default_loss_mask_path)
                    item_info.audio_loss_mask_intervals = loss_mask_intervals
                    data.append(item_info)
                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= self.batch_size or (len(data) > 0 and flush):
                batch = data[0 : self.batch_size]
                if len(data) > self.batch_size:
                    data = data[self.batch_size :]
                else:
                    data = []
                return batch
            return None

        for fetch_op in self.datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                if self._uses_ltx2_audio_video_geometry():
                    yield (int(self.resolution[0]), int(self.resolution[1])), batch
                else:
                    yield (1, 1), batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            if self._uses_ltx2_audio_video_geometry():
                yield (int(self.resolution[0]), int(self.resolution[1])), batch
            else:
                yield (1, 1), batch
        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_text_encoder_output_cache_batches is not available when cache_only=True")
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        audio_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_audio.safetensors"))
        bucketed_item_info: dict[tuple[int, int], list[ItemInfo]] = {}

        for audio_cache_file in audio_cache_files:
            base = os.path.basename(audio_cache_file)
            suffix = f"_{self.architecture}_audio.safetensors"
            if not base.endswith(suffix):
                continue
            if self._uses_ltx2_audio_video_geometry():
                latent_cache_file = os.path.join(
                    self.cache_directory,
                    base[: -len(suffix)] + f"_{self.architecture}.safetensors",
                )
                if not os.path.exists(latent_cache_file):
                    logger.warning(f"Video latent cache file not found: {latent_cache_file}")
                    continue

                latent_stem = os.path.basename(latent_cache_file)[: -len(f"_{self.architecture}.safetensors")]
                original_size = (int(self.resolution[0]), int(self.resolution[1]))
                item_key = latent_stem
                if "_" in latent_stem:
                    key_stem, resolution_token = latent_stem.rsplit("_", 1)
                    if "x" in resolution_token:
                        w_s, h_s = resolution_token.split("x", 1)
                        try:
                            original_size = (int(w_s), int(h_s))
                            item_key = key_stem
                        except ValueError:
                            item_key = latent_stem

                text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
                if not os.path.exists(text_encoder_output_cache_file):
                    logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                    continue

                bucket_reso = self._append_audio_bucket_key((original_size[0], original_size[1]), True)
                item_info = ItemInfo(item_key, "", original_size, bucket_reso, latent_cache_path=latent_cache_file)
                item_info.text_encoder_output_cache_path = text_encoder_output_cache_file
                item_info.audio_latent_cache_path = audio_cache_file
            else:
                item_key = self._legacy_strip_resolution_suffix(base[: -len(suffix)])
                latent_cache_file = os.path.join(self.cache_directory, f"{item_key}_0001x0001_{self.architecture}.safetensors")
                if not os.path.exists(latent_cache_file):
                    logger.warning(f"Video latent cache file not found: {latent_cache_file}")
                    continue
                text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
                if not os.path.exists(text_encoder_output_cache_file):
                    logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                    continue

                bucket_reso = self._append_audio_bucket_key((1, 1), True)
                item_info = ItemInfo(item_key, "", (1, 1), bucket_reso, latent_cache_path=latent_cache_file)
                item_info.text_encoder_output_cache_path = text_encoder_output_cache_file
                item_info.audio_latent_cache_path = audio_cache_file

            # Duration bucketing: group audio clips by quantized length to minimize
            # padding within batches.  Reads only the safetensors header (fast).
            # Convert audio_bucket_interval (seconds) to latent frames (25 fps).
            _AUDIO_DURATION_BUCKET_STEP = max(int(round(self.audio_bucket_interval * 25)), 1)
            audio_key = safetensors_utils.find_key(audio_cache_file, starts_with="audio_latents_")
            if audio_key is not None:
                try:
                    # key format: audio_latents_{T}x{F}x{C}_{dtype}
                    dims_part = audio_key.split("_")[2]  # "{T}x{F}x{C}"
                    audio_t = int(dims_part.split("x")[0])
                    if self.audio_bucket_strategy == "truncate":
                        # Floor division: all items in bucket have T >= quantized_t
                        quantized_t = max(
                            (audio_t // _AUDIO_DURATION_BUCKET_STEP) * _AUDIO_DURATION_BUCKET_STEP,
                            _AUDIO_DURATION_BUCKET_STEP,
                        )
                    else:
                        # Round-to-nearest (pad mode)
                        quantized_t = max(
                            ((audio_t + _AUDIO_DURATION_BUCKET_STEP // 2) // _AUDIO_DURATION_BUCKET_STEP)
                            * _AUDIO_DURATION_BUCKET_STEP,
                            _AUDIO_DURATION_BUCKET_STEP,
                        )
                    bucket_reso = (*bucket_reso, quantized_t)
                except (ValueError, IndexError):
                    pass

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        target_fps = 24.0
        if self.architecture in {ARCHITECTURE_LTX2, ARCHITECTURE_LTX2_FULL}:
            target_fps = VideoDataset.TARGET_FPS_LTX2

        self.batch_manager = BucketBatchManager(
            bucketed_item_info,
            self.batch_size,
            num_timestep_buckets=num_timestep_buckets,
            architecture=self.architecture,
            target_fps=target_fps,
            audio_bucket_strategy=self.audio_bucket_strategy,
            video_loss_weight=self.video_loss_weight,
            audio_loss_weight=self.audio_loss_weight,
        )
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


class VideoDataset(BaseDataset):
    TARGET_FPS_HUNYUAN = 24.0
    TARGET_FPS_WAN = 16.0
    TARGET_FPS_LTX2 = 25.0
    TARGET_FPS_FRAMEPACK = 30.0
    TARGET_FPS_FLUX_KONTEXT = 1.0  # VideoDataset is not used for Flux Kontext, but this is a placeholder
    TARGET_FPS_HUNYUAN_VIDEO_1_5 = 24.0

    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        video_loss_weight: Optional[float] = None,
        audio_loss_weight: Optional[float] = None,
        caption_field: Optional[str] = None,
        frame_extraction: Optional[str] = "head",
        frame_stride: Optional[int] = 1,
        frame_sample: Optional[int] = 1,
        target_frames: Optional[list[int]] = None,
        max_frames: Optional[int] = None,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        video_directory: Optional[str] = None,
        video_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        reference_directory: Optional[str] = None,
        reference_directories: Optional[Sequence[str]] = None,
        reference_audio_directory: Optional[str] = None,
        reference_audio_directories: Optional[Sequence[str]] = None,
        cache_directory: Optional[str] = None,
        reference_cache_directory: Optional[str] = None,
        reference_cache_directories: Optional[Sequence[str]] = None,
        reference_audio_cache_directory: Optional[str] = None,
        reference_audio_cache_directories: Optional[Sequence[str]] = None,
        separate_audio_buckets: bool = False,
        loss_mask_directory: Optional[str] = None,
        default_loss_mask_path: Optional[str] = None,
        loss_mask_use_alpha: bool = False,
        loss_mask_invert: bool = False,
        fp_latent_window_size: Optional[int] = 9,
        cache_only: bool = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        super(VideoDataset, self).__init__(
            resolution,
            caption_extension,
            caption_field,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            video_loss_weight,
            audio_loss_weight,
            cache_directory,
            reference_cache_directory,
            reference_cache_directories,
            reference_audio_cache_directory,
            reference_audio_cache_directories,
            separate_audio_buckets,
            loss_mask_directory,
            default_loss_mask_path,
            loss_mask_use_alpha,
            loss_mask_invert,
            debug_dataset,
            architecture,
        )
        self.video_directory = video_directory
        self.video_jsonl_file = video_jsonl_file
        self.control_directory = control_directory
        self.reference_directories = _normalize_optional_path_list(reference_directory, reference_directories)
        self.reference_directory = self.reference_directories[0] if self.reference_directories else None
        self.reference_audio_directories = _normalize_optional_path_list(
            reference_audio_directory,
            reference_audio_directories,
        )
        self.reference_audio_directory = (
            self.reference_audio_directories[0] if self.reference_audio_directories else None
        )
        self.frame_extraction = frame_extraction
        self.frame_stride = frame_stride
        self.frame_sample = frame_sample
        self.max_frames = max_frames
        self.source_fps = source_fps
        self.fp_latent_window_size = fp_latent_window_size
        self.cache_only = cache_only

        self.vae_frame_stride = 4  # all architectures require frames to be divisible by 4
        if self.architecture == ARCHITECTURE_HUNYUAN_VIDEO:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_WAN:
            self.target_fps = VideoDataset.TARGET_FPS_WAN
        elif self.architecture == ARCHITECTURE_LTX2:
            self.target_fps = target_fps if target_fps is not None else VideoDataset.TARGET_FPS_LTX2
        elif self.architecture == ARCHITECTURE_FRAMEPACK:
            self.target_fps = VideoDataset.TARGET_FPS_FRAMEPACK
        elif self.architecture == ARCHITECTURE_FLUX_KONTEXT:
            self.target_fps = VideoDataset.TARGET_FPS_FLUX_KONTEXT
        elif self.architecture == ARCHITECTURE_KANDINSKY5:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_HUNYUAN_VIDEO_1_5:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN_VIDEO_1_5
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        if target_frames is not None:
            target_frames = list(set(target_frames))
            target_frames.sort()

            # round each value to N*4+1
            rounded_target_frames = [(f - 1) // self.vae_frame_stride * self.vae_frame_stride + 1 for f in target_frames]
            rounded_target_frames = list(set(rounded_target_frames))
            rounded_target_frames.sort()

            # if value is changed, warn
            if target_frames != rounded_target_frames:
                logger.warning(f"target_frames are rounded to {rounded_target_frames}")

            target_frames = tuple(rounded_target_frames)

        self.target_frames = target_frames

        if self.cache_only:
            self.datasource = None
        elif video_directory is not None:
            self.datasource = VideoDirectoryDatasource(
                video_directory,
                caption_extension,
                control_directory,
                loss_mask_directory=loss_mask_directory,
                loss_mask_invert=loss_mask_invert,
            )
        elif video_jsonl_file is not None:
            self.datasource = VideoJsonlDatasource(
                video_jsonl_file,
                caption_field=caption_field,
                loss_mask_directory=loss_mask_directory,
                loss_mask_invert=loss_mask_invert,
            )
        else:
            raise ValueError("video_directory or video_jsonl_file must be specified")

        if not self.cache_only and self.frame_extraction == "uniform" and self.frame_sample == 1:
            self.frame_extraction = "head"
            logger.warning("frame_sample is set to 1 for frame_extraction=uniform. frame_extraction is changed to head.")
        if not self.cache_only and self.frame_extraction == "head":
            # head extraction. we can limit the number of frames to be extracted
            self.datasource.set_start_and_end_frame(0, max(self.target_frames))

        if self.cache_directory is None:
            self.cache_directory = self.video_directory
        if self.cache_only and self.cache_directory is None:
            raise ValueError("cache_directory is required when cache_only=True")

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control if self.datasource is not None else False

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.video_directory is not None:
            metadata["video_directory"] = os.path.basename(self.video_directory)
        if self.video_jsonl_file is not None:
            metadata["video_jsonl_file"] = os.path.basename(self.video_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["frame_extraction"] = self.frame_extraction
        metadata["frame_stride"] = self.frame_stride
        metadata["frame_sample"] = self.frame_sample
        metadata["target_frames"] = self.target_frames
        metadata["max_frames"] = self.max_frames
        metadata["source_fps"] = self.source_fps
        metadata["has_control"] = self.has_control
        metadata["cache_only"] = self.cache_only
        return metadata

    def retrieve_latent_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_latent_cache_batches is not available when cache_only=True")
        buckset_selector = BucketSelector(
            self.resolution,
            architecture=self.architecture,
            reference_downscale=getattr(self, "reference_downscale", 1),
        )
        self.datasource.set_bucket_selector(buckset_selector)
        self.datasource.set_source_and_target_fps(self.source_fps, self.target_fps)

        executor = ThreadPoolExecutor(max_workers=num_workers)

        # key: (width, height, frame_count) and optional latent_window_size, value: [ItemInfo]
        batches: dict[tuple[Any], list[ItemInfo]] = {}
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_frame_size, video_key, video, caption, control, loss_mask = future.result()

                    frame_count = len(video)
                    video = np.stack(video, axis=0)
                    height, width = video.shape[1:3]
                    bucket_reso = (width, height)  # already resized

                    # process control images if available
                    control_video = None
                    if control is not None:
                        # set frame count to the same as video
                        if len(control) > frame_count:
                            control = control[:frame_count]
                        elif len(control) < frame_count:
                            # if control is shorter than video, repeat the last frame
                            last_frame = control[-1]
                            control.extend([last_frame] * (frame_count - len(control)))
                        control_video = np.stack(control, axis=0)

                    loss_mask_video = None
                    if loss_mask is not None:
                        loss_mask_video = np.asarray(loss_mask, dtype=np.float32)

                    crop_pos_and_frames = []
                    if self.frame_extraction == "head":
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                crop_pos_and_frames.append((0, target_frame))
                    elif self.frame_extraction == "chunk":
                        # split by target_frames
                        for target_frame in self.target_frames:
                            for i in range(0, frame_count, target_frame):
                                if i + target_frame <= frame_count:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "slide":
                        # slide window
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                for i in range(0, frame_count - target_frame + 1, self.frame_stride):
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "uniform":
                        # select N frames uniformly
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                frame_indices = np.linspace(0, frame_count - target_frame, self.frame_sample, dtype=int)
                                for i in frame_indices:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "full":
                        # select all frames
                        target_frame = min(frame_count, self.max_frames)
                        target_frame = (target_frame - 1) // self.vae_frame_stride * self.vae_frame_stride + 1  # round to N*4+1
                        crop_pos_and_frames.append((0, target_frame))
                    else:
                        raise ValueError(f"frame_extraction {self.frame_extraction} is not supported")

                    for crop_pos, target_frame in crop_pos_and_frames:
                        cropped_video = video[crop_pos : crop_pos + target_frame]
                        body, ext = os.path.splitext(video_key)
                        item_key = f"{body}_{crop_pos:05d}-{target_frame:03d}{ext}"
                        batch_key = (*bucket_reso, target_frame)  # bucket_reso with frame_count

                        if self.architecture == ARCHITECTURE_FRAMEPACK:
                            # add latent window size to bucket resolution
                            batch_key = (*batch_key, self.fp_latent_window_size)

                        # crop control video if available
                        cropped_control = None
                        if control_video is not None:
                            cropped_control = control_video[crop_pos : crop_pos + target_frame]

                        cropped_loss_mask = None
                        if loss_mask_video is not None:
                            cropped_loss_mask = loss_mask_video[crop_pos : crop_pos + target_frame]

                        item_info = ItemInfo(
                            item_key, caption, original_frame_size, batch_key, frame_count=target_frame, content=cropped_video
                        )
                        item_info.source_item_key = video_key
                        item_info.source_total_frames = frame_count
                        item_info.chunk_start_frame = crop_pos
                        item_info.chunk_num_frames = target_frame
                        item_info.latent_cache_path = self.get_latent_cache_path(item_info)

                        if self.reference_cache_directories:
                            item_info.reference_latent_cache_paths = self.get_reference_latent_cache_paths(item_info)
                            item_info.reference_latent_cache_path = item_info.reference_latent_cache_paths[0]
                        if self.reference_audio_cache_directories:
                            item_info.reference_audio_latent_cache_paths = self.get_reference_audio_latent_cache_paths(item_info)
                            item_info.reference_audio_latent_cache_path = item_info.reference_audio_latent_cache_paths[0]
                        item_info.control_content = cropped_control  # None is allowed
                        item_info.loss_mask_content = cropped_loss_mask
                        item_info.fp_latent_window_size = self.fp_latent_window_size

                        batch = batches.get(batch_key, [])
                        batch.append(item_info)
                        batches[batch_key] = batch

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for operator in self.datasource:

            def fetch_and_resize(op: callable) -> tuple[tuple[int, int], str, list[np.ndarray], str, Optional[list[np.ndarray]], Optional[np.ndarray]]:
                result = op()

                if len(result) == 3:  # for backward compatibility TODO remove this in the future
                    video_key, video, caption = result
                    control = None
                    loss_mask = None
                elif len(result) == 4:
                    video_key, video, caption, control = result
                    loss_mask = None
                else:
                    video_key, video, caption, control, loss_mask = result

                video: list[np.ndarray]
                frame_size = (video[0].shape[1], video[0].shape[0])

                # resize if necessary
                bucket_reso = buckset_selector.get_bucket_resolution(frame_size)
                video = [resize_image_to_bucket(frame, bucket_reso) for frame in video]

                # resize control if necessary
                if control is not None:
                    control = [resize_image_to_bucket(frame, bucket_reso) for frame in control]

                resized_loss_mask = None
                if loss_mask is not None:
                    resized_loss_mask = np.stack(
                        [loss_mask_to_float_array(frame, bucket_reso) for frame in loss_mask],
                        axis=0,
                    )
                elif self.default_loss_mask_path:
                    resized_loss_mask = load_loss_mask_frames(
                        self.default_loss_mask_path,
                        bucket_reso=bucket_reso,
                        frame_count=len(video),
                        source_fps=self.source_fps,
                        target_fps=self.target_fps,
                        invert=self.loss_mask_invert,
                    )

                return frame_size, video_key, video, caption, control, resized_loss_mask

            future = executor.submit(fetch_and_resize, operator)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        if self.datasource is None:
            raise ValueError("retrieve_text_encoder_output_cache_batches is not available when cache_only=True")
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        bucket_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
            self.architecture,
            reference_downscale=getattr(self, "reference_downscale", 1),
        )

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int, int], list[ItemInfo]] = {}  # (width, height, frame_count) -> [ItemInfo]
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            frame_pos, frame_count = tokens[-3].split("-")[:2]  # "00000-000", or optional section index "00000-000-00"
            frame_pos, frame_count = int(frame_pos), int(frame_count)

            item_key = "_".join(tokens[:-3])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)
            bucket_reso = (*bucket_reso, frame_count)
            audio_latent_cache_file = self.get_audio_latent_cache_path_from_latent_cache_path(cache_file)
            has_audio = os.path.exists(audio_latent_cache_file)
            bucket_reso = self._append_audio_bucket_key(tuple(bucket_reso), has_audio)
            item_info = ItemInfo(item_key, "", image_size, bucket_reso, frame_count=frame_count, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file
            item_info.audio_latent_cache_path = audio_latent_cache_file if has_audio else None

            dino_cache_file = self.get_dino_feature_cache_path_from_latent_cache_path(cache_file)
            item_info.dino_feature_cache_path = dino_cache_file if os.path.exists(dino_cache_file) else None

            if self.reference_cache_directories:
                reference_cache_paths: list[str] = []
                missing_reference_cache = False
                for reference_cache_directory in self.reference_cache_directories:
                    ref_cache_path = os.path.join(reference_cache_directory, os.path.basename(cache_file))
                    if os.path.exists(ref_cache_path):
                        reference_cache_paths.append(ref_cache_path)
                    else:
                        logger.warning(f"Reference cache not found, skipping item: {ref_cache_path}")
                        missing_reference_cache = True
                        break
                if missing_reference_cache:
                    continue
                if reference_cache_paths:
                    item_info.reference_latent_cache_paths = reference_cache_paths
                    item_info.reference_latent_cache_path = reference_cache_paths[0]
            if self.reference_audio_cache_directories:
                reference_audio_cache_paths: list[str] = []
                missing_reference_audio_cache = False
                for reference_audio_cache_directory in self.reference_audio_cache_directories:
                    ref_audio_cache_path = os.path.join(
                        reference_audio_cache_directory,
                        os.path.basename(cache_file).replace(
                            f"_{self.architecture}.safetensors",
                            f"_{self.architecture}_audio.safetensors",
                        ),
                    )
                    if os.path.exists(ref_audio_cache_path):
                        reference_audio_cache_paths.append(ref_audio_cache_path)
                    else:
                        logger.warning(f"Reference audio cache not found, skipping item: {ref_audio_cache_path}")
                        missing_reference_audio_cache = True
                        break
                if missing_reference_audio_cache:
                    continue
                if reference_audio_cache_paths:
                    item_info.reference_audio_latent_cache_paths = reference_audio_cache_paths
                    item_info.reference_audio_latent_cache_path = reference_audio_cache_paths[0]

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(
            bucketed_item_info,
            self.batch_size,
            num_timestep_buckets=num_timestep_buckets,
            architecture=self.architecture,
            target_fps=self.target_fps,
            video_loss_weight=self.video_loss_weight,
            audio_loss_weight=self.audio_loss_weight,
        )
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[ImageDataset, VideoDataset, AudioDataset]]):
        super().__init__(datasets)
        self.datasets: list[Union[ImageDataset, VideoDataset, AudioDataset]] = datasets
        self.num_train_items = 0
        for dataset in self.datasets:
            self.num_train_items += dataset.num_train_items

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

