from __future__ import annotations

import glob
from importlib.util import find_spec
import os
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from PIL import Image
import cv2
import av

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF"]


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
            caption_base = os.path.splitext(os.path.basename(caption_path))[0]
            caption_bases.add(caption_base)
        filtered_img_paths = []
        for img_path in img_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            if img_base in caption_bases:
                filtered_img_paths.append(img_path)
        img_paths = filtered_img_paths

    img_paths.sort()
    return img_paths


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


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


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
    if source_fps is None or target_fps is None:
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


def _get_video_fps(container, fallback_fps: Optional[float]) -> float:
    if fallback_fps is not None and fallback_fps > 0:
        return float(fallback_fps)
    video_streams = [stream for stream in container.streams if stream.type == "video"]
    if video_streams:
        rate = video_streams[0].average_rate or video_streams[0].base_rate
        if rate is not None:
            return float(rate)
    raise ValueError("Could not determine video FPS for audio crop alignment. Pass source_fps/target_fps in dataset config.")


def _audio_frame_to_channels_first(frame: av.AudioFrame, channels: int) -> np.ndarray:
    array = frame.to_ndarray()
    if array.ndim == 1:
        array = array[None, :]
    elif array.shape[0] != channels and array.shape[-1] == channels:
        array = array.T

    if array.shape[0] == 1 and channels == 2:
        array = np.repeat(array, 2, axis=0)
    elif array.shape[0] > channels:
        array = array[:channels]
    elif array.shape[0] < channels:
        pad = np.zeros((channels - array.shape[0], array.shape[1]), dtype=array.dtype)
        array = np.concatenate([array, pad], axis=0)
    return array.astype(np.float32, copy=False)


def load_audio(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    sample_rate: int = 48000,
    channels: int = 2,
    video_fps: Optional[float] = None,
) -> np.ndarray:
    """Load audio from a video file as float32 [channels, samples].

    ``start_frame`` and ``end_frame`` use the same frame coordinates as the
    video crop. ``video_fps`` should be the FPS used for those coordinates.
    """
    if not os.path.isfile(video_path):
        raise ValueError(f"Audio extraction requires a video file, got directory: {video_path}")

    container = av.open(video_path)
    try:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            raise ValueError(f"No audio stream found in video: {video_path}")

        crop_fps = _get_video_fps(container, video_fps)
        start_sample = 0 if start_frame is None else int(round(start_frame / crop_fps * sample_rate))
        end_sample = None if end_frame is None else int(round(end_frame / crop_fps * sample_rate))
        target_samples = None if end_sample is None else max(0, end_sample - start_sample)

        layout = "stereo" if channels == 2 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(_audio_frame_to_channels_first(resampled, channels))

        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            for resampled in flushed:
                chunks.append(_audio_frame_to_channels_first(resampled, channels))

        if chunks:
            audio = np.concatenate(chunks, axis=1)
        else:
            audio = np.zeros((channels, 0), dtype=np.float32)

        if start_sample > audio.shape[1]:
            audio = np.zeros((channels, 0), dtype=np.float32)
        else:
            audio = audio[:, start_sample:end_sample]

        if target_samples is not None and audio.shape[1] < target_samples:
            pad = np.zeros((channels, target_samples - audio.shape[1]), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=1)
        return audio
    finally:
        container.close()
