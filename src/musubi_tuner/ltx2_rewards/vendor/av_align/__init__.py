"""Vendored AV-Align peak-detection primitives (algorithmic, no model/checkpoint).

Implements the AV-Align metric: detect audio onset peaks + video optical-flow
peaks and compute the Intersection-over-Union of their times; higher = better
synchronization.

This is a pure-algorithm reward: there is no neural model and no checkpoint, so the
whole thing is self-contained code. Public functions mirror the original:

  - ``detect_audio_peaks``        — spectral-flux onset detection -> peak times (s)
  - ``extract_frames``            — OpenCV frame decode (+ optional resize)
  - ``detect_video_peaks``        — Farneback optical-flow magnitude -> local-max times
  - ``calc_intersection_over_union`` — IoU of audio vs video peak times

Dependency surface (lazy-imported inside the functions so the registry and the
deterministic CPU IoU-math tests never need media tooling):

  - numpy            (IoU math; always)
  - opencv (cv2)     (``extract_frames`` / ``detect_video_peaks``)
  - librosa          (``detect_audio_peaks`` EXACT primary: onset_strength + onset_detect)
  - torch / torchaudio (``detect_audio_peaks`` FALLBACK: wav decode + mel spectrogram + STFT)

PARITY NOTE — librosa = EXACT primary, torchaudio = APPROXIMATE fallback
------------------------------------------------------------------------
``detect_audio_peaks`` is structured as **exact-primary + lightweight-fallback**:

  - **librosa present (primary):** the function runs the canonical recipe —
    ``librosa.load`` -> ``librosa.onset.onset_strength`` -> ``librosa.onset.onset_detect``
    -> ``librosa.frames_to_time``. This is BIT-IDENTICAL to the reference AV-Align
    metric (IoU parity to 1e-12).
  - **librosa absent (fallback):** the function falls back to a torch/torchaudio/numpy
    reimplementation — a mel-spectrogram spectral-flux onset envelope followed by an
    adaptive local-max peak picker (Böck et al., the heuristic librosa itself uses) —
    with matched default parameters (``sr=22050``, ``n_fft=2048``, ``hop_length=512``,
    ``n_mels=128``, ``delta=0.07`` and ~30 ms / ~100 ms peak-pick windows). AV-Align is
    an *algorithmic* AV-sync heuristic (not a learned model), so this fallback is parity
    APPROXIMATE — the onset *count*/*times* can differ slightly (different STFT
    windowing/edge handling, mel filterbank normalization, float paths) and the absolute
    IoU can shift — but it remains a faithful spectral-flux onset detector that still
    discriminates in-sync from out-of-sync audio/video, which is what the reward needs.

``tqdm`` is optional (only the interactive progress bar); ``detect_video_peaks`` is
always called with ``use_tqdm=False`` from the reward path.
"""

from __future__ import annotations


# resize frames
def resize_frames(frames, new_size_scheme):
    """
    Args:
        frames (list): the elements in frames are numpy.ndarray.
        new_size_scheme (str):  resize scheme.
    Return:
        frames: the elements in list are resized frames.
    """
    import cv2

    h, w, _ = frames[0].shape
    # new_w, new_h = w, h
    if new_size_scheme.startswith("min"):
        min_edge = int(new_size_scheme.split("=")[1])
        scale_ratio = min_edge / min(w, h)
        new_h = int(scale_ratio * h)
        new_w = int(scale_ratio * w)
    elif new_size_scheme.find(":") != -1:
        new_w = int(new_size_scheme.split(":")[0])
        new_h = int(new_size_scheme.split(":")[1])

    if (w, h) == (new_w, new_h):
        return frames

    new_frames = []
    for img in frames:
        new_frames.append(cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR))

    return new_frames


# Function to extract frames from a video file
def extract_frames(video_path, resize_scheme=None, max_length_s=None):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        frames (list): List of frames extracted from the video.
        frame_rate (float): Frame rate of the video.
    """
    import cv2

    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open the video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_length_s is not None and len(frames) >= frame_rate * max_length_s:
            break
    cap.release()
    if resize_scheme is not None:
        frames = resize_frames(frames, resize_scheme)
    return frames, frame_rate


# Default analysis parameters (mirror the librosa onset-detection defaults).
_AUDIO_SR = 22050  # librosa.load default resample target
_N_FFT = 2048
_HOP_LENGTH = 512
_N_MELS = 128


def _load_wav_pcm(path):
    """Read a 16-bit PCM wav with the stdlib (no codec backend). Returns ``([ch, samples] f32, sr)``."""
    import wave

    import numpy as np
    import torch

    with wave.open(path, "rb") as w:
        file_sr, channels, nframes = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(nframes)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    arr = arr.reshape(-1, channels).T if channels > 1 else arr.reshape(1, -1)
    return torch.from_numpy(arr.copy()), file_sr


def _load_audio_mono(audio_file, sr=None, max_length_s=None):
    """Load an audio file to a mono float32 numpy waveform at the target ``sr``.

    Uses ``torchaudio`` (which handles wav/flac/mp3/... via its backends). Channels
    are averaged to mono and the signal is resampled to ``sr`` (default 22050,
    matching ``librosa.load``). Returns ``(y, sr)`` where ``y`` is 1-D float32.
    """
    import torch
    import torchaudio

    target_sr = _AUDIO_SR if sr is None else int(sr)

    try:
        waveform, file_sr = torchaudio.load(audio_file)  # (channels, samples), float32
    except Exception:
        # torchaudio>=2.9 routes load() through torchcodec, which may be absent. The RL rollout
        # writes a standard 16-bit PCM wav, so fall back to a stdlib reader (no codec backend).
        waveform, file_sr = _load_wav_pcm(audio_file)
    # Downmix to mono.
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to the target sample rate (no-op when already matching).
    if file_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, target_sr)

    if max_length_s is not None:
        max_samples = int(max_length_s * target_sr)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

    y = waveform.squeeze(0).to(torch.float32).contiguous()
    return y, target_sr


def _onset_strength_envelope(y, sr):
    """Spectral-flux onset-strength envelope from a mel spectrogram (torch).

    Follows librosa's ``onset_strength`` recipe: power mel spectrogram -> dB,
    half-wave-rectified frame-to-frame difference (positive energy increase), then
    mean-aggregate across mel bands. Returns a 1-D float32 numpy array, one value
    per STFT frame (``center=True``, ``hop_length=_HOP_LENGTH``).
    """
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sr),
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        n_mels=_N_MELS,
        power=2.0,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    )
    S = mel(y.unsqueeze(0)).squeeze(0)  # (n_mels, n_frames)
    # Power -> dB (reference = max power), matching librosa.power_to_db default top_db=80.
    ref = S.max().clamp_min(1e-10)
    S_db = 10.0 * torch.log10(S.clamp_min(1e-10) / ref)
    S_db = S_db.clamp_min(S_db.max() - 80.0)

    # Spectral flux: half-wave-rectified first difference along time (lag=1).
    diff = S_db[:, 1:] - S_db[:, :-1]
    diff = torch.clamp(diff, min=0.0)
    # Pad the leading frame so the envelope length matches the number of frames.
    onset_env = torch.zeros(S_db.shape[1], dtype=torch.float32)
    onset_env[1:] = diff.mean(dim=0)
    return onset_env.cpu().numpy()


def _peak_pick(onset_env, sr, hop_length=_HOP_LENGTH):
    """Adaptive local-max peak picker (Böck et al. heuristic, librosa-compatible).

    Sample ``n`` is a peak iff:
      1. ``x[n] == max(x[n - pre_max : n + post_max])``
      2. ``x[n] >= mean(x[n - pre_avg : n + post_avg]) + delta``
      3. ``n - last_peak > wait``
    with windows/threshold derived from the same defaults librosa uses. The envelope
    is normalized to its max first (``normalize=True``). Returns frame indices.
    """
    import numpy as np

    x = np.asarray(onset_env, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=int)

    # Normalize to [0, 1] (librosa onset_detect normalize=True).
    peak = x.max()
    if peak > 0:
        x = x / peak

    fps = float(sr) / float(hop_length)
    # Time-constant windows mirroring librosa.onset.onset_detect defaults.
    pre_max = max(1, int(round(0.03 * fps)))
    post_max = max(1, int(round(0.00 * fps)) + 1)
    pre_avg = max(1, int(round(0.10 * fps)))
    post_avg = max(1, int(round(0.10 * fps)) + 1)
    wait = max(1, int(round(0.03 * fps)))
    delta = 0.07

    n = x.shape[0]
    peaks = []
    last = None
    for i in range(n):
        lo_max = max(0, i - pre_max)
        hi_max = min(n, i + post_max)
        if x[i] != x[lo_max:hi_max].max():
            continue
        lo_avg = max(0, i - pre_avg)
        hi_avg = min(n, i + post_avg)
        if x[i] < x[lo_avg:hi_avg].mean() + delta:
            continue
        if last is not None and i - last <= wait:
            continue
        peaks.append(i)
        last = i

    return np.asarray(peaks, dtype=int)


# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file=None, y=None, sr=None, max_length_s=None):
    """
    Detect audio peaks using the Onset Detection algorithm.

    EXACT primary path: ``librosa`` (``onset_strength`` + ``onset_detect`` +
    ``frames_to_time``) — bit-identical to the reference AV-Align metric. If ``librosa``
    is not installed, falls back to a torch/torchaudio/numpy spectral-flux detector
    (``_detect_audio_peaks_torchaudio``) whose parity is APPROXIMATE. See the module
    docstring for details.

    Args:
        audio_file (str): Path to the audio file.
        y: Optional pre-loaded mono waveform (numpy array or torch tensor).
        sr: Sample rate (required if ``y`` is provided; defaults to 22050 on load).
        max_length_s: Optional cap on analyzed audio length, in seconds.

    Returns:
        onset_times (np.ndarray): Times (in seconds) where audio peaks occur.
    """
    try:
        import librosa  # EXACT primary path (reference-identical)
        import numpy as _np
    except ImportError:
        # librosa absent -> torch/torchaudio approximate fallback.
        return _detect_audio_peaks_torchaudio(audio_file=audio_file, y=y, sr=sr, max_length_s=max_length_s)

    # --- VERBATIM reference librosa onset detection (exact parity) ---
    if y is None:
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        assert y is not None and sr is not None
        # librosa requires a numpy float array; coerce a passed-in torch tensor / list
        # (the file-loaded path already yields numpy from librosa.load). This is a no-op
        # for a numpy float32 array, so bit-identical parity is preserved.
        if not isinstance(y, _np.ndarray):
            try:
                import torch as _torch

                if isinstance(y, _torch.Tensor):
                    y = y.detach().cpu().numpy()
            except ImportError:
                pass
            y = _np.asarray(y, dtype=_np.float32)
    if max_length_s is not None and len(y) > max_length_s * sr:
        y = y[: int(max_length_s * sr)]
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def _detect_audio_peaks_torchaudio(audio_file=None, y=None, sr=None, max_length_s=None):
    """torch/torchaudio/numpy spectral-flux onset fallback (librosa-free).

    Used only when ``librosa`` cannot be imported. Parity vs the librosa primary is
    APPROXIMATE, not bit-identical (see the module docstring). Same signature/semantics
    as ``detect_audio_peaks``.
    """
    import numpy as np
    import torch

    if y is None:
        y, sr = _load_audio_mono(audio_file, sr=sr, max_length_s=max_length_s)
    else:
        assert y is not None and sr is not None
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        y = y.to(torch.float32).reshape(-1)
        if max_length_s is not None and y.shape[0] > max_length_s * sr:
            y = y[: int(max_length_s * sr)]

    # Calculate the onset envelope (spectral flux over a mel spectrogram).
    onset_env = _onset_strength_envelope(y, sr)
    # Get the onset events (adaptive local-max peak picking).
    onset_frames = _peak_pick(onset_env, sr, hop_length=_HOP_LENGTH)
    # Convert frame indices to times in seconds (librosa.frames_to_time equivalent).
    onset_times = onset_frames.astype(np.float64) * (_HOP_LENGTH / float(sr))
    return onset_times


# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """
    local_extrema_indexes = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:  # Local maximum
            local_extrema_indexes.append(i / fps)

    return local_extrema_indexes


# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps, use_tqdm=True):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    if len(frames) == 0:
        return None, []

    if isinstance(frames[0], float):
        return None, frames

    # flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    flow_trajectory = [compute_of(frames[0], frames[1])]
    pbar = range(1, len(frames))
    if use_tqdm:
        from tqdm import tqdm

        pbar = tqdm(pbar, desc="Process Frames")
    for i in pbar:
        flow_trajectory.append(compute_of(frames[i - 1], frames[i]))

    video_peaks = find_local_max_indexes(flow_trajectory, fps)

    return flow_trajectory, video_peaks


# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    import cv2

    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    intersection_length = 0
    for audio_peak in audio_peaks:
        for video_peak in video_peaks:
            if video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                break

    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length + 1e-6)


__all__ = [
    "resize_frames",
    "extract_frames",
    "detect_audio_peaks",
    "find_local_max_indexes",
    "detect_video_peaks",
    "compute_of",
    "calc_intersection_over_union",
]
