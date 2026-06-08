#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Vendored from facebookresearch/ImageBind.
# Inference-only. Deviations from upstream:
#   - ``from imagebind.models.* import ...`` -> relative imports.
#   - ``return_bpe_path`` resolves the bundled bpe vocab relative to this package
#     instead of using ``pkg_resources.resource_filename`` (no setuptools dependency).
#   - ``load_and_transform_video_data`` is structured as EXACT-primary + fallback:
#       * ``pytorchvideo`` present (PRIMARY): the VERBATIM upstream ImageBind
#         decoder (``EncodedVideo`` + ``ConstantClipsPerVideoSampler`` +
#         ``pv_transforms``) -> bit-identical upstream video embeddings.
#       * ``pytorchvideo`` absent (FALLBACK): decode with ``torchvision.io.read_video``
#         (PyAV fallback if the torchvision codec backend is missing) and reimplement
#         clip/frame sampling in pure torch -> parity-APPROXIMATE video embeddings.
#     The audio clip-timepoint math is always reproduced exactly via
#     ``constant_clips_per_video_timepoints`` (used by the audio path and the torchvision
#     video fallback); the text/image/audio paths are unaffected and remain exact.

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from .models.multimodal_preprocessors import SimpleTokenizer

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = os.path.join(os.path.dirname(__file__), "bpe", "bpe_simple_vocab_16e6.txt.gz")


def return_bpe_path():
    return BPE_PATH


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and target_length (%d). Is the audio_target_length setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def constant_clips_per_video_timepoints(clip_duration, clips_per_video, duration):
    """Pure-python reimplementation of pytorchvideo ``ConstantClipsPerVideoSampler``.

    Returns ``clips_per_video`` evenly-spaced ``(start, end)`` windows of length
    ``clip_duration`` (seconds) spanning ``[0, duration]``. This reproduces the upstream
    sampler's math exactly (``augs_per_clip=1``): clip ``i`` starts at
    ``i * (max(duration - clip_duration, 0) / max(clips_per_video - 1, 1))`` and ends at
    ``start + clip_duration``. Used for both the audio (mel) and video (frame) paths so
    their clip *timepoints* stay bit-identical to upstream.
    """
    max_possible_clip_start = max(duration - clip_duration, 0)
    uniform_clip = max_possible_clip_start / max(clips_per_video - 1, 1)
    all_clips_timepoints = []
    for clip_index in range(clips_per_video):
        start = uniform_clip * clip_index
        all_clips_timepoints.append((start, start + clip_duration))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=return_bpe_path())
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        all_clips_timepoints = constant_clips_per_video_timepoints(clip_duration, clips_per_video, waveform.size(1) / sample_rate)
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(clip_timepoints[1] * sample_rate),
            ]
            waveform_melspec = waveform2melspec(waveform_clip, sample_rate, num_mel_bins, target_length)
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


def uniform_temporal_subsample(x, num_samples):
    """Reimplementation of pytorchvideo ``UniformTemporalSubsample`` (temporal dim=1).

    Uniformly subsamples ``num_samples`` indices from the temporal axis of a
    ``[C, T, H, W]`` tensor. Reproduces pytorchvideo's index math exactly:
    ``torch.linspace(0, t - 1, num_samples)`` clamped to ``[0, t - 1]`` and cast to long,
    then ``index_select`` along the temporal dim.
    """
    t = x.shape[1]
    indices = torch.linspace(0, t - 1, num_samples, device=x.device)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, 1, indices)


def short_side_scale(x, size, interpolation="bilinear"):
    """Reimplementation of pytorchvideo ``ShortSideScale`` for a ``[C, T, H, W]`` tensor.

    Resizes so the short spatial side equals ``size`` while preserving aspect ratio,
    using the same default bilinear interpolation (``align_corners=False``) as upstream.
    """
    c, t, h, w = x.shape
    if w < h:
        new_w = size
        new_h = int(math.floor((float(h) / w) * size))
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return F.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)


def _decode_video_frames(video_path):
    """Decode a video file to ``(frames[T, H, W, C] uint8, fps)``.

    Primary path: ``torchvision.io.read_video``. Newer torchvision builds route this
    through TorchCodec, which may be unavailable; in that case we fall back to decoding
    with PyAV directly (the same backend pytorchvideo used). Either way the returned
    pixels feed the identical downstream sampling/scale/crop/normalize, so the choice of
    decoder is the only source of approximate (vs pytorchvideo) parity.
    """
    try:
        frames, _audio, info = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")
        fps = info.get("video_fps", None)
        if frames.shape[0] > 0 and fps:
            return frames, float(fps)
    except Exception:  # pragma: no cover - depends on the torchvision codec backend
        pass

    # PyAV fallback (decode all frames as RGB24).
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    avg_rate = stream.average_rate or stream.base_rate
    fps = float(avg_rate) if avg_rate else 0.0
    decoded = []
    for frame in container.decode(video=0):
        decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    container.close()
    if not decoded:
        raise ValueError(f"No frames decoded from {video_path!r}")
    frames = torch.stack(decoded, dim=0)  # [T, H, W, C] uint8
    if not fps or fps <= 0:
        raise ValueError(f"Could not determine video fps for {video_path!r}")
    return frames, fps


def get_clip_timepoints(clip_sampler, duration):
    """Read out all ``(start, end)`` clip timepoints from a pytorchvideo clip sampler.

    Verbatim from upstream ImageBind ``data.py`` — used only by the pytorchvideo
    EXACT primary video path below.
    """
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    """Load video files, sample clips, and transform to ImageBind's expected tensor.

    EXACT-primary + fallback dispatcher:

      - ``pytorchvideo`` installed (PRIMARY): runs the VERBATIM upstream ImageBind
        video transform (``EncodedVideo`` + ``ConstantClipsPerVideoSampler`` +
        ``pv_transforms.ShortSideScale`` / ``UniformTemporalSubsample``). Video embeddings
        are bit-identical to upstream ImageBind.
      - ``pytorchvideo`` absent (FALLBACK): decodes with ``torchvision.io.read_video``
        (PyAV fallback) and reimplements clip/frame sampling in pure torch. Video
        embeddings are parity-APPROXIMATE (different decoder; see the fallback docstring).

    The text/image/audio paths are unaffected by either branch and remain exact.
    """
    if video_paths is None:
        return None

    try:
        import pytorchvideo  # noqa: F401  (presence check for the exact primary path)
    except ImportError:
        return _load_and_transform_video_data_torchvision(
            video_paths,
            device,
            clip_duration=clip_duration,
            clips_per_video=clips_per_video,
            sample_rate=sample_rate,
        )

    return _load_and_transform_video_data_pytorchvideo(
        video_paths,
        device,
        clip_duration=clip_duration,
        clips_per_video=clips_per_video,
        sample_rate=sample_rate,
    )


def _ensure_pytorchvideo_torchvision_compat():
    """Make ``pytorchvideo<=0.1.5`` importable on ``torchvision>=0.17``.

    pytorchvideo's ``transforms`` package eagerly imports
    ``torchvision.transforms.functional_tensor``, a module removed in torchvision 0.17
    (its public symbols were merged into ``torchvision.transforms.functional``). We alias
    the old name to the current ``functional`` module so the VERBATIM upstream transforms
    import cleanly. This only runs when the pytorchvideo primary path is selected and only
    touches this process's ``sys.modules`` — it does not modify the installed venv.
    """
    import sys

    import torchvision.transforms.functional as _F

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = _F
    import torchvision.transforms as _tvt

    if not hasattr(_tvt, "functional_tensor"):
        _tvt.functional_tensor = _F


def _load_and_transform_video_data_pytorchvideo(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    """VERBATIM upstream ImageBind pytorchvideo video transform (EXACT primary).

    Copied from upstream ImageBind ``data.py:load_and_transform_video_data``
    (decode_audio=False). Decoding stays inside pytorchvideo's ``EncodedVideo`` machinery,
    so the sampled/scaled/cropped frames — and thus the video embedding — are bit-identical
    to upstream ImageBind for a given decoder.

    Two env-driven compatibility notes vs the literal upstream call (neither changes the
    decoded pixels relative to upstream-with-the-same-decoder):
      - Upstream passed ``decoder="decord"``. We use decord only if it's importable and
        otherwise pytorchvideo's native ``"pyav"`` decoder (pytorchvideo's own default),
        because decord ships no wheels for this torch/python and the upstream ``decord``
        decoder is itself a thin pytorchvideo wrapper.
      - Upstream passed ``sample_rate=`` to ``from_path``; pytorchvideo 0.1.5's signature
        does not accept it (it only affects audio, and we decode video-only), so it is
        dropped.
    """
    _ensure_pytorchvideo_torchvision_compat()

    from pytorchvideo import transforms as pv_transforms
    from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
    from pytorchvideo.data.encoded_video import EncodedVideo

    # Prefer the upstream decord decoder when available; else pytorchvideo's native pyav.
    try:
        import decord  # noqa: F401

        _decoder = "decord"
    except ImportError:
        _decoder = "pyav"

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=clips_per_video)
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decoder=_decoder,
            decode_audio=False,
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)


def _load_and_transform_video_data_torchvision(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    """Load video files, sample clips, and transform to ImageBind's expected tensor.

    PARITY NOTE: upstream ImageBind decodes/samples video frames with ``pytorchvideo``
    (``EncodedVideo`` + ``ConstantClipsPerVideoSampler`` + ``UniformTemporalSubsample``).
    This FALLBACK (used only when ``pytorchvideo`` is not installed) instead decodes with
    ``torchvision.io.read_video`` (falling back to PyAV if the torchvision codec backend is
    unavailable; see ``_decode_video_frames``) and reimplements the clip/frame sampling in
    pure torch. The clip *timepoints* and the short-side-scale / spatial-crop / normalize
    steps are reproduced exactly, but the underlying video decoder differs (torchvision/PyAV
    vs pytorchvideo's decord/pyav), so the exact decoded pixels — and therefore the resulting
    video embedding — are APPROXIMATE, not bit-identical, relative to the pytorchvideo path.
    The text/image/audio paths are unaffected and remain exact.
    """
    if video_paths is None:
        return None

    video_outputs = []
    # ShortSideScale(224) is applied per-clip below; this Compose only normalizes.
    normalize = NormalizeVideo(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    for video_path in video_paths:
        # Decode the full video to a [T, H, W, C] uint8 tensor + fps (pts in seconds).
        frames, fps = _decode_video_frames(video_path)
        num_frames = frames.shape[0]
        duration = num_frames / fps

        # [T, H, W, C] uint8 -> [C, T, H, W] float (0-255); matches pytorchvideo layout.
        all_frames = frames.permute(3, 0, 1, 2).float()

        all_clips_timepoints = constant_clips_per_video_timepoints(clip_duration, clips_per_video, duration)

        all_video = []
        for clip_start, clip_end in all_clips_timepoints:
            start_idx = int(math.floor(clip_start * fps))
            # pytorchvideo get_clip is inclusive of the end timepoint's frame.
            end_idx = int(math.floor(clip_end * fps))
            start_idx = max(0, min(start_idx, num_frames - 1))
            end_idx = max(start_idx + 1, min(end_idx + 1, num_frames))
            clip = all_frames[:, start_idx:end_idx, :, :]
            if clip.shape[1] == 0:
                raise ValueError("No clip found")

            video_clip = uniform_temporal_subsample(clip, num_samples=clip_duration)
            video_clip = video_clip / 255.0  # since this is float, need 0-1
            video_clip = short_side_scale(video_clip, 224)
            video_clip = normalize(video_clip)
            all_video.append(video_clip)

        all_video = SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)
