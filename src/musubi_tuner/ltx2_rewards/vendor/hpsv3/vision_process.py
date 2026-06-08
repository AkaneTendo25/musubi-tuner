"""Vision preprocessing for the HPSv3 inferencer (image path only).

Vendored from github.com/MizzenAI/HPSv3 (hpsv3/dataset/utils.py),
itself adapted from qwen-vl-utils. The external hpsv3 package uses THIS local
copy (not qwen_vl_utils) in inference.py, so the inferencer here uses it too to
stay bit-identical. The image helpers are copied VERBATIM; the video readers and
their packaging import are dropped (the inferencer only passes image entries).
"""

from __future__ import annotations

import base64
import math
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    if isinstance(image_obj, Image.Image):
        image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        if isinstance(image, torch.Tensor):
            shape = image.shape
            if len(shape) == 4:
                if shape[1] in [1, 3]:  # Likely [B, C, H, W]
                    height, width = shape[2], shape[3]
                    image_mode = "NCHW"
                elif shape[3] in [1, 3]:  # Likely [B, H, W, C]
                    height, width = shape[1], shape[2]
                    image_mode = "NHWC"

            elif len(shape) == 3:
                if shape[0] in [1, 3]:  # Likely [C, H, W]
                    height, width = shape[1], shape[2]
                    image_mode = "CHW"
                elif shape[2] in [1, 3]:  # Likely [H, W, C]
                    height, width = shape[0], shape[1]
                    image_mode = "HWC"
                else:
                    raise ValueError(f"Cannot determine tensor image format from shape {shape}")
            else:
                raise ValueError(f"Unsupported tensor image shape: {shape}")
        else:
            width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    if isinstance(image, torch.Tensor):
        if image_mode == "NCHW":
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == "NHWC":
            image = transforms.functional.resize(
                image.permute(0, 3, 1, 2), [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == "CHW":
            image = image.unsqueeze(0)  # Add batch dimension
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension and change to CHW
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )

    else:
        # If the image is a PIL Image, we resize it using PIL.
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((resized_width, resized_height), Image.BICUBIC)

    return image


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "image" in ele or "image_url" in ele or "video" in ele or ele["type"] in ("image", "image_url", "video"):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


def fetch_video(ele, image_factor=IMAGE_FACTOR):
    # Video reading (decord/torchvision) is intentionally not vendored: the HPSv3
    # inferencer only feeds still images. Kept as a stub so the verbatim
    # process_vision_info above has a defined symbol; raises if ever reached.
    raise NotImplementedError("vendored hpsv3.vision_process supports image inputs only; got a 'video' entry")
