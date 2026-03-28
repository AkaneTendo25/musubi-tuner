"""
VACE latent caching for LTX-2 training.

Encodes VACE control videos (+ masks) and saves the resulting context tensors
alongside the main video latent caches. Follows the same pattern as
encode_and_save_reference_latents in ltx2_cache_latents.py.

Supports all VACE task types by accepting pre-prepared control videos and masks
in the vace_directory. Expected file structure:

    vace_directory/
        video_name.mp4          # control video (depth, pose, source for inpaint, etc.)
        video_name_mask.mp4     # mask video (optional; white=reactive, black=inactive)
        video_name_mask.png     # or single mask image applied to all frames

If no mask file is found, defaults to all-ones mask (everything reactive).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import numpy as np
import torch

from musubi_tuner.dataset.image_video_dataset import (
    BaseDataset,
    IMAGE_EXTENSIONS,
    ItemInfo,
    VIDEO_EXTENSIONS,
    VideoDataset,
    save_latent_cache_ltx2,
    resize_image_to_bucket,
)
from musubi_tuner.ltx_vace.vace_control_encoder import (
    LTX2_VAE_SPATIAL_COMPRESSION,
    prepare_vace_context,
)

logger = logging.getLogger(__name__)


def _find_vace_file(vace_directory: str, stem: str) -> Optional[str]:
    """Find a VACE control video/image matching the given stem."""
    all_exts = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
    for ext in all_exts:
        candidate = os.path.join(vace_directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    # Case-insensitive fallback
    lower_stem = stem.lower()
    try:
        for fname in os.listdir(vace_directory):
            name_no_ext, ext = os.path.splitext(fname)
            if name_no_ext.lower() == lower_stem and ext.lower() in [e.lower() for e in all_exts]:
                return os.path.join(vace_directory, fname)
    except OSError:
        pass
    return None


def _find_mask_file(vace_directory: str, stem: str) -> Optional[str]:
    """Find a mask file for the given stem (e.g., video_name_mask.mp4 or .png)."""
    mask_stem = f"{stem}_mask"
    all_exts = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
    for ext in all_exts:
        candidate = os.path.join(vace_directory, mask_stem + ext)
        if os.path.exists(candidate):
            return candidate
    # Also check without _mask suffix with .mask. infix
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(vace_directory, f"{stem}.mask{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_vace_frames(
    path: str,
    bucket_reso: tuple[int, int],
    num_frames: int,
) -> np.ndarray:
    """Load control/mask video frames as [F, H, W, 3] uint8."""
    from PIL import Image
    import av

    ext = os.path.splitext(path)[1].lower()
    is_video = ext in [e.lower() for e in VIDEO_EXTENSIONS]

    if is_video:
        container = av.open(path)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i >= num_frames:
                break
            pil_frame = frame.to_image()
            arr = resize_image_to_bucket(pil_frame, bucket_reso)
            frames.append(arr)
        container.close()
        if not frames:
            raise ValueError(f"No frames decoded from VACE video: {path}")
        # Pad to num_frames if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
    else:
        image = Image.open(path).convert("RGB")
        arr = resize_image_to_bucket(image, bucket_reso)
        frames = [arr] * num_frames

    return np.stack(frames[:num_frames], axis=0)


def _load_mask_frames(
    path: str,
    bucket_reso: tuple[int, int],
    num_frames: int,
) -> np.ndarray:
    """Load mask frames as [F, H, W, 1] float32 (0 or 1)."""
    from PIL import Image
    import av

    ext = os.path.splitext(path)[1].lower()
    is_video = ext in [e.lower() for e in VIDEO_EXTENSIONS]

    if is_video:
        container = av.open(path)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i >= num_frames:
                break
            pil_frame = frame.to_image().convert("L")
            arr = resize_image_to_bucket(pil_frame, bucket_reso)
            if arr.ndim == 3:
                arr = arr[:, :, 0:1]
            else:
                arr = arr[:, :, np.newaxis]
            frames.append(arr)
        container.close()
        while len(frames) < num_frames:
            frames.append(frames[-1])
    else:
        image = Image.open(path).convert("L")
        arr = resize_image_to_bucket(image, bucket_reso)
        if arr.ndim == 3:
            arr = arr[:, :, 0:1]
        else:
            arr = arr[:, :, np.newaxis]
        frames = [arr] * num_frames

    mask = np.stack(frames[:num_frames], axis=0).astype(np.float32) / 255.0
    # Binarize: > 0.5 = 1 (reactive), <= 0.5 = 0 (inactive)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def encode_and_save_vace_latents(
    vae,
    datasets: Sequence[BaseDataset],
    args: argparse.Namespace,
    device: torch.device,
    tiling_config=None,
) -> None:
    """Encode VACE control videos + masks and save as latent caches.

    For each training video, looks for a matching control video (and optional mask)
    in the vace_directory. Encodes them and saves the VACE context tensor.
    """
    skip_existing = getattr(args, "skip_existing", False)
    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)

    # Mask augmentation (applied during caching for offline augmentation)
    mask_augmentor = None
    if getattr(args, "vace_mask_augmentation", False):
        from musubi_tuner.ltx_vace.mask_augmentation import MaskAugmentor
        mask_augmentor = MaskAugmentor()
        logger.info("VACE mask augmentation enabled during caching")

    vae_param = next(vae.parameters())
    vae_dtype = vae_param.dtype

    for ds_idx, ds in enumerate(datasets):
        if not isinstance(ds, VideoDataset):
            continue

        vace_cache_dir = getattr(ds, "vace_cache_directory", None)
        if vace_cache_dir is None:
            logger.info(f"[Dataset {ds_idx}] No vace_cache_directory set, skipping VACE caching")
            continue

        vace_dir = getattr(ds, "vace_directory", None)
        if vace_dir is None:
            logger.info(f"[Dataset {ds_idx}] No vace_directory set, skipping VACE caching")
            continue

        # Warn about temporal misalignment for non-head frame extraction
        frame_extraction = getattr(ds, "frame_extraction", "head")
        if frame_extraction not in ("head", None):
            logger.warning(
                "[Dataset %d] VACE caching with frame_extraction='%s': control video frames "
                "are always loaded from the start, which may not align with training clips "
                "extracted from later positions. Consider using frame_extraction='head' for "
                "VACE training or preparing per-clip control videos.",
                ds_idx, frame_extraction,
            )

        os.makedirs(vace_cache_dir, exist_ok=True)
        logger.info(f"[Dataset {ds_idx}] Caching VACE latents to {vace_cache_dir}")

        cached_count = 0
        skipped_count = 0
        missing_count = 0

        for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
            for item_info in batch:
                vace_cache_path = getattr(item_info, "vace_latent_cache_path", None)
                if vace_cache_path is None:
                    vace_cache_path = ds.get_vace_latent_cache_path(item_info)

                if skip_existing and os.path.exists(vace_cache_path):
                    skipped_count += 1
                    continue

                source_key = getattr(item_info, "source_item_key", None) or item_info.item_key
                stem = os.path.splitext(os.path.basename(source_key))[0]
                bucket_reso = (item_info.bucket_size[0], item_info.bucket_size[1])
                num_frames = getattr(item_info, "frame_count", 1) or 1

                try:
                    # Find control video
                    vace_path = _find_vace_file(vace_dir, stem)
                    if vace_path is None:
                        missing_count += 1
                        if missing_count <= 5:
                            logger.warning(f"No VACE control file found for '{stem}' in {vace_dir}")
                        elif missing_count == 6:
                            logger.warning("(suppressing further missing-VACE warnings)")
                        continue

                    # Load control frames
                    control_frames = _load_vace_frames(vace_path, bucket_reso, num_frames)
                    # (F, H, W, 3) uint8 -> (1, 3, F, H, W) float [-1, 1]
                    control_tensor = torch.from_numpy(control_frames).unsqueeze(0)
                    control_tensor = control_tensor.permute(0, 4, 1, 2, 3).contiguous()
                    control_tensor = control_tensor.to(device=device, dtype=vae_dtype)
                    control_tensor = control_tensor / 127.5 - 1.0

                    # Load mask (or default to all-ones)
                    mask_path = _find_mask_file(vace_dir, stem)
                    if mask_path is not None:
                        mask_frames = _load_mask_frames(mask_path, bucket_reso, num_frames)
                        # (F, H, W, 1) -> (1, 1, F, H, W)
                        mask_tensor = torch.from_numpy(mask_frames).unsqueeze(0)
                        mask_tensor = mask_tensor.permute(0, 4, 1, 2, 3).contiguous()
                        mask_tensor = mask_tensor.to(device=device, dtype=vae_dtype)
                    else:
                        # Default: all-ones mask (everything reactive)
                        mask_tensor = torch.ones(
                            1, 1, control_tensor.shape[2], control_tensor.shape[3], control_tensor.shape[4],
                            device=device, dtype=vae_dtype,
                        )

                    # Apply mask augmentation if enabled
                    if mask_augmentor is not None:
                        mask_tensor = mask_augmentor(mask_tensor)

                    # Temporal padding for VAE (frames must be 1 mod 8)
                    frames = control_tensor.shape[2]
                    remainder = (frames - 1) % 8
                    if remainder != 0:
                        pad = 8 - remainder
                        control_tensor = torch.cat(
                            [control_tensor, control_tensor[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2
                        )
                        mask_tensor = torch.cat(
                            [mask_tensor, mask_tensor[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2
                        )

                    # Load reference image if available (for R2V tasks)
                    # Looks for {stem}_ref.png/jpg or {stem}_ref.mp4
                    ref_tensor = None
                    ref_stem = f"{stem}_ref"
                    ref_path = _find_vace_file(vace_dir, ref_stem)
                    if ref_path is not None:
                        ref_frames = _load_vace_frames(ref_path, bucket_reso, 1)  # single ref frame
                        ref_tensor = torch.from_numpy(ref_frames).unsqueeze(0)
                        ref_tensor = ref_tensor.permute(0, 4, 1, 2, 3).contiguous()
                        ref_tensor = ref_tensor.to(device=device, dtype=vae_dtype)
                        ref_tensor = ref_tensor / 127.5 - 1.0

                    # Encode via VAE using prepare_vace_context
                    def _vae_encode(x):
                        with torch.no_grad():
                            if tiling_config is not None and hasattr(vae, "tiled_encode"):
                                return vae.tiled_encode(x, tiling_config).to(device=device, dtype=vae_dtype)
                            return vae(x).to(device=device, dtype=vae_dtype)

                    with torch.no_grad():
                        vace_context = prepare_vace_context(
                            control_video=control_tensor,
                            mask=mask_tensor,
                            vae_encode_fn=_vae_encode,
                            reference_images=ref_tensor,
                            device=device,
                            dtype=vae_dtype,
                        )

                    # Save as safetensors with vace_ prefix
                    vace_latent = vace_context[0]  # Remove batch dim
                    vace_item_info = ItemInfo(
                        item_info.item_key,
                        item_info.caption,
                        item_info.original_size,
                        item_info.bucket_size,
                    )
                    vace_item_info.latent_cache_path = vace_cache_path
                    vace_item_info.frame_count = num_frames
                    save_latent_cache_ltx2(vace_item_info, vace_latent)
                    cached_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cache VACE for '{stem}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        logger.info(
            f"[Dataset {ds_idx}] VACE caching done: {cached_count} cached, "
            f"{skipped_count} skipped (existing), {missing_count} missing"
        )


# =====================================================================
# Audio VACE latent caching
# =====================================================================

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"]


def _find_audio_vace_file(audio_vace_directory: str, stem: str) -> Optional[str]:
    """Find an audio VACE control file matching the given stem."""
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(audio_vace_directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    # Case-insensitive fallback
    lower_stem = stem.lower()
    try:
        for fname in os.listdir(audio_vace_directory):
            name_no_ext, ext = os.path.splitext(fname)
            if name_no_ext.lower() == lower_stem and ext.lower() in [e.lower() for e in AUDIO_EXTENSIONS]:
                return os.path.join(audio_vace_directory, fname)
    except OSError:
        pass
    return None


def _find_audio_mask_file(audio_vace_directory: str, stem: str) -> Optional[str]:
    """Find an audio mask file (e.g., stem_mask.wav or stem_mask.npy)."""
    mask_stem = f"{stem}_mask"
    # Check for numpy mask
    npy_path = os.path.join(audio_vace_directory, mask_stem + ".npy")
    if os.path.exists(npy_path):
        return npy_path
    # Check for audio mask files
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(audio_vace_directory, mask_stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def encode_and_save_audio_vace_latents(
    audio_encoder,
    audio_processor,
    datasets: Sequence[BaseDataset],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Encode audio VACE control signals + masks and save as latent caches.

    Uses the same mel preprocessing pipeline as the main audio latent caching:
    load waveform -> normalize to stereo -> waveform_to_mel -> encoder(mel).

    Saved keys:
        ``latents_{T}x{F}_{dtype}`` — control audio latents ``[C, T, F]``
        ``audio_vace_mask``          — temporal mask ``[T]`` (1 = generate)

    Args:
        audio_encoder: Audio VAE encoder (expects mel spectrogram input).
        audio_processor: AudioProcessor with ``waveform_to_mel()`` method.
    """
    import torchaudio
    from safetensors.torch import save_file as st_save

    skip_existing = getattr(args, "skip_existing", False)

    encoder_param = next(audio_encoder.parameters())
    encoder_dtype = encoder_param.dtype

    for ds_idx, ds in enumerate(datasets):
        if not isinstance(ds, VideoDataset):
            continue

        audio_vace_cache_dir = getattr(ds, "audio_vace_cache_directory", None)
        if audio_vace_cache_dir is None:
            continue

        audio_vace_dir = getattr(ds, "audio_vace_directory", None)
        if audio_vace_dir is None:
            logger.info(f"[Dataset {ds_idx}] No audio_vace_directory set, skipping audio VACE caching")
            continue

        frame_extraction = getattr(ds, "frame_extraction", "head")
        if frame_extraction not in ("head", None):
            logger.warning(
                "[Dataset %d] Audio VACE caching with frame_extraction='%s': entire control "
                "audio is encoded, which may not align with training clips from later "
                "positions. Consider per-clip control audio files.",
                ds_idx, frame_extraction,
            )

        os.makedirs(audio_vace_cache_dir, exist_ok=True)
        logger.info(f"[Dataset {ds_idx}] Caching audio VACE latents to {audio_vace_cache_dir}")

        cached_count = 0
        skipped_count = 0
        missing_count = 0

        for _bucket_key, batch in ds.retrieve_latent_cache_batches(1):
            for item_info in batch:
                audio_vace_cache_path = getattr(item_info, "audio_vace_latent_cache_path", None)
                if audio_vace_cache_path is None:
                    audio_vace_cache_path = ds.get_audio_vace_latent_cache_path(item_info)

                if skip_existing and os.path.exists(audio_vace_cache_path):
                    skipped_count += 1
                    continue

                source_key = getattr(item_info, "source_item_key", None) or item_info.item_key
                stem = os.path.splitext(os.path.basename(source_key))[0]

                try:
                    # Find control audio
                    control_audio_path = _find_audio_vace_file(audio_vace_dir, stem)
                    if control_audio_path is None:
                        missing_count += 1
                        if missing_count <= 5:
                            logger.warning(f"No audio VACE control file found for '{stem}' in {audio_vace_dir}")
                        elif missing_count == 6:
                            logger.warning("(suppressing further missing audio VACE warnings)")
                        continue

                    # Load waveform (same pipeline as ltx2_cache_latents.py)
                    waveform, sample_rate = torchaudio.load(control_audio_path)

                    # Normalize to stereo (audio encoder expects 2 channels)
                    channels = int(waveform.shape[0])
                    if channels == 1:
                        waveform = waveform.repeat(2, 1)
                    elif channels > 2:
                        mono = waveform.float().mean(dim=0, keepdim=True)
                        waveform = mono.repeat(2, 1)

                    waveform = waveform.unsqueeze(0)  # (1, 2, samples)

                    # Compute mel spectrogram in fp32, then encode
                    waveform = waveform.to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        mel = audio_processor.waveform_to_mel(waveform, int(sample_rate))
                        mel = mel.to(device=device, dtype=encoder_dtype)
                        control_latents = audio_encoder(mel)
                    # control_latents: (1, C, T, F)

                    latents_3d = control_latents[0].detach().cpu().contiguous()  # (C, T, F)
                    T = int(latents_3d.shape[1])

                    # Build temporal mask (T,)
                    mask_path = _find_audio_mask_file(audio_vace_dir, stem)
                    if mask_path is not None and mask_path.endswith(".npy"):
                        mask_np = np.load(mask_path).astype(np.float32)
                        if mask_np.ndim > 1:
                            mask_np = mask_np.mean(axis=tuple(range(1, mask_np.ndim)))
                        mask_1d = torch.from_numpy(mask_np)
                        if mask_1d.shape[0] != T:
                            import torch.nn.functional as TF
                            mask_1d = TF.interpolate(
                                mask_1d.view(1, 1, -1), size=T, mode="nearest-exact"
                            ).view(T)
                        mask_1d = (mask_1d > 0.5).float()
                    elif mask_path is not None:
                        # Audio mask file — encode, derive temporal mask from energy
                        try:
                            mask_wav, mask_sr = torchaudio.load(mask_path)
                            if mask_wav.shape[0] == 1:
                                mask_wav = mask_wav.repeat(2, 1)
                            elif mask_wav.shape[0] > 2:
                                mask_wav = mask_wav.float().mean(dim=0, keepdim=True).repeat(2, 1)
                            mask_wav = mask_wav.unsqueeze(0).to(device=device, dtype=torch.float32)
                            with torch.no_grad():
                                mask_mel = audio_processor.waveform_to_mel(mask_wav, int(mask_sr))
                                mask_mel = mask_mel.to(device=device, dtype=encoder_dtype)
                                mask_latents = audio_encoder(mask_mel)
                            energy_per_t = mask_latents[0].abs().mean(dim=(0, 2))  # (T,)
                            mask_1d = (energy_per_t > energy_per_t.median() * 0.1).float().cpu()
                            if mask_1d.shape[0] != T:
                                import torch.nn.functional as TF
                                mask_1d = TF.interpolate(
                                    mask_1d.view(1, 1, -1), size=T, mode="nearest-exact"
                                ).view(T)
                        except Exception as e:
                            logger.warning(f"Failed to load audio mask '{mask_path}': {e}, using all-ones")
                            mask_1d = torch.ones(T, dtype=torch.float32)
                    else:
                        mask_1d = torch.ones(T, dtype=torch.float32)

                    # Save latents + mask
                    dtype_str = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
                        latents_3d.dtype, str(latents_3d.dtype)
                    )
                    C_raw, T_lat, F_lat = latents_3d.shape
                    sd = {
                        f"latents_{T_lat}x{F_lat}_{dtype_str}": latents_3d,
                        "audio_vace_mask": mask_1d.detach().cpu().contiguous(),
                    }
                    st_save(sd, audio_vace_cache_path)
                    cached_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cache audio VACE for '{stem}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        logger.info(
            f"[Dataset {ds_idx}] Audio VACE caching done: {cached_count} cached, "
            f"{skipped_count} skipped (existing), {missing_count} missing"
        )


def encode_and_save_joint_av_vace_latents(
    vae,
    audio_encoder,
    audio_processor,
    datasets: Sequence[BaseDataset],
    args: argparse.Namespace,
    device: torch.device,
    tiling_config=None,
) -> None:
    """Cache both video and audio VACE latents sequentially.

    Delegates to encode_and_save_vace_latents() and
    encode_and_save_audio_vace_latents() for the actual encoding.
    Note: each function loads masks independently; temporal alignment
    between video and audio masks is not enforced by this function.
    For aligned masks, prepare matching mask files in both directories.
    """
    # First, cache video VACE latents (existing function handles this)
    encode_and_save_vace_latents(vae, datasets, args, device, tiling_config)

    # Then, cache audio VACE latents
    encode_and_save_audio_vace_latents(audio_encoder, audio_processor, datasets, args, device)

    # Log alignment info for datasets that have both
    for ds_idx, ds in enumerate(datasets):
        if not isinstance(ds, VideoDataset):
            continue
        has_video_vace = getattr(ds, "vace_cache_directory", None) is not None
        has_audio_vace = getattr(ds, "audio_vace_cache_directory", None) is not None
        if has_video_vace and has_audio_vace:
            logger.info(
                "[Dataset %d] Joint AV VACE caching complete. Both video and audio "
                "VACE caches are available.",
                ds_idx,
            )
