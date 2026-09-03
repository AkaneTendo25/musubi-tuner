from __future__ import annotations

import argparse
import bisect
import copy
import gc
import hashlib
import json
import logging
import os
import math
import re
import time
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from multiprocessing import Value
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner import convert_lora
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.hv_train import get_sigmas
from musubi_tuner.hv_train_network import NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_h3.architecture import (
    AUDIO_FLOW_SHIFT,
    AUDIO_LATENT_FPS,
    VIDEO_DIT_PATCH_SIZE,
    VIDEO_FLOW_SHIFT,
    VIDEO_FPS,
    VIDEO_LATENT_CHANNELS,
    align_frame_count,
    temporal_shape,
)
from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import (
    H3PairedConditioningUnsupportedError,
    H3TrainingBackend,
    create_conditioning_encoder,
    create_training_backend,
)
from musubi_tuner.minimax_h3.block_sparse_attention import DEFAULT_BLOCK
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_DOP_CONFIG_KEY,
    H3_DOP_TEXT_HIDDEN_KEY,
    H3_DOP_TEXT_TOKEN_TAGS_KEY,
    H3_REFERENCE_MODALITY_PROBABILITIES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
)
from musubi_tuner.minimax_h3.dop import dop_config_identity
from musubi_tuner.minimax_h3.component_loader import load_audio_vae_decoder, load_video_vae_decoder
from musubi_tuner.minimax_h3.crepa import H3CREPA, H3CREPAConfig, parse_crepa_config
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3.inference import (
    decode_latents_sequentially,
    denoise_fl2va,
    encode_keyframe_images,
    prepare_keyframe_image,
    save_av_mp4,
)
from musubi_tuner.minimax_h3.masking import (
    CONDITIONING_MASK_BATCH_KEY as H3_CONDITIONING_MASK_KEY,
)
from musubi_tuner.minimax_h3.masking import (
    audio_mask_to_rows,
    rows_to_latent_video_mask,
    sample_audio_mask,
    sample_video_mask,
    video_mask_to_rows,
)
from musubi_tuner.minimax_h3.model import h3_profile_scope
from musubi_tuner.minimax_h3.packing import AUDIO_CHANNELS, MiniMaxH3GuideGeometry
from musubi_tuner.minimax_h3.rollout import (
    MAX_ROLLOUT_WINDOW,
    SIGMA_FLOOR as ROLLOUT_SIGMA_FLOOR,
    TEACHER_PRIVILEGE_CHANNELS,
    H3RolloutTeacherCache,
    batch_item_key,
    enable_item_keys,
    euler_advance,
    item_key_latent_caches,
    item_key_text_caches,
    rollout_base_sigmas,
    teacher_batch,
)
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODES,
    REFERENCE_VIDEO_FPS,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
    validate_reference_video_fps,
    validate_reference_video_sizing,
)
from musubi_tuner.minimax_h3.training import (
    H3FusedArm,
    H3JointNoisyInputs,
    H3ModelPrediction,
    cfg_zero_rescaled_empty,
    contrastive_guidance_target,
    guidance_consistent_prediction,
    guidance_scale_for_sigma,
    joint_prediction_loss,
    joint_velocity_loss,
    prepare_joint_noisy_inputs,
    shift_sigma,
    unshift_sigma,
)
from musubi_tuner.minimax_h3.validation import (
    H3ValidationAccumulator,
    image_validation_sigma,
    masked_squared_error_sum,
    preserve_rng_state,
    seed_validation_forward,
    validation_sigma_bins,
)
from musubi_tuner.training.accelerator_setup import collator_class
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import LOSS_FOR_AVERAGE_KEY
from musubi_tuner.training.validation import derive_validation_seed
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

_SAMPLE_KEYFRAME_ROWS = "_h3_keyframe_rows"
_SAMPLE_KEYFRAME_ANCHORS = "_h3_keyframe_anchors"

_DIRECT_SIGMA_SAMPLING = {
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "krea2_shift",
    "ideogram4_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
}

_H3_BASE_TIMESTEP_SAMPLING = {"sigma", "uniform", "sigmoid", "shift", "logsnr"}

_H3_LORA_TARGETS = ("attention", "mlp", "audio", "video", "token_refiner")
_H3_TRANSFORMER_BLOCKS = 50

# Kernel-name patterns that sort profiler rows into the categories of the ``--h3_profile_steps``
# table. Attention is matched first: cutlass-built fused attention kernels ("fmha_cutlassF...")
# also match the GEMM pattern and belong to attention, not to the projections around it. GEMM
# covers cuBLAS on Hopper ("nvjet_tst_...", "sm90_xmma_gemm_..."), Ampere ("ampere_bf16_...gemm",
# "gemmSN_..."), cutlass, gemv and hipBLASLt ("Cijk_..."); attention covers FA2/FA3 ("flash_fwd_kernel",
# "flash::...", "flash_bwd..."), cutlass/cuDNN fused attention and this repo's int8 and block-sparse
# kernels. The profile file lists the top kernels by name so a miss can be fixed from the artifact.
H3_PROFILE_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "attention",
        re.compile(r"flash|fmha|fwd_kernel|bwd_kernel|cudnn.*attn|sdp|attention|block_sparse", re.IGNORECASE),
    ),
    ("gemm", re.compile(r"gemm|xmma|cutlass|matmul|nvjet|gemv|Cijk_", re.IGNORECASE)),
    ("swap", re.compile(r"Memcpy HtoD|Memcpy DtoH")),
)
H3_PROFILE_OTHER = "elementwise/other"
H3_PROFILE_IDLE = "idle"
H3_PROFILE_TOP_KERNELS = 30
H3_PROFILE_WAIT_STEPS = 2
H3_PROFILE_WARMUP_STEPS = 1


def h3_profile_category(name: str) -> str:
    for candidate, pattern in H3_PROFILE_CATEGORIES:
        if pattern.search(name):
            return candidate
    return H3_PROFILE_OTHER


def categorize_h3_profile_rows(rows) -> dict[str, tuple[float, int]]:
    """Aggregate ``(kernel_name, device_time_us[, launches])`` rows into ``{category: (time_us, kernels)}``.

    Every category of ``H3_PROFILE_CATEGORIES`` plus ``H3_PROFILE_OTHER`` is present in the
    result, in that order, even when it received nothing. The kernel count is of distinct rows."""
    totals: dict[str, list] = {name: [0.0, 0] for name, _ in H3_PROFILE_CATEGORIES}
    totals[H3_PROFILE_OTHER] = [0.0, 0]
    for row in rows:
        name, time_us = row[0], row[1]
        category = h3_profile_category(name)
        totals[category][0] += float(time_us)
        totals[category][1] += 1
    return {name: (time_us, count) for name, (time_us, count) in totals.items()}


def collect_h3_profile_rows(events, device_type: str) -> tuple[list[tuple[str, float, int]], float]:
    """Reduce a profiler's ``events()`` to ``(name, time_us, launches)`` kernel rows and the busy time.

    Only leaf device kernels count. ``record_function`` labels and the ``ProfilerStep#`` markers
    show up on the device timeline as *user annotations* spanning every kernel beneath them, and
    CPU-side ops (``aten::mm``, autograd nodes) carry the device time of the kernels they launch:
    either would count each kernel several times over, which is why ``key_averages()`` is not
    used here. Rows come back sorted by time, descending; the busy time is the union of the
    kernel intervals (``device_busy_us``). Without a device the ops' self CPU time stands in."""
    cuda_type = getattr(torch.autograd.DeviceType, "CUDA", None)
    totals: dict[str, list] = {}
    intervals = []
    for event in events:
        if getattr(event, "is_user_annotation", False) or getattr(event, "is_async", False):
            continue
        name = str(event.name)
        if name.startswith("ProfilerStep#"):
            continue
        time_range = getattr(event, "time_range", None)
        if device_type == "cuda":
            if getattr(event, "device_type", None) != cuda_type or time_range is None:
                continue
            time_us = float(time_range.end) - float(time_range.start)
        else:
            time_us = float(event.self_cpu_time_total)
        if time_range is not None:
            intervals.append((time_range.start, time_range.end))
        entry = totals.setdefault(name, [0.0, 0])
        entry[0] += time_us
        entry[1] += 1
    rows = sorted(((name, time_us, count) for name, (time_us, count) in totals.items()), key=lambda row: -row[1])
    return rows, device_busy_us(intervals)


def format_h3_profile_top_kernels(rows, limit: int = H3_PROFILE_TOP_KERNELS) -> str:
    """List the heaviest kernels with the category each was filed under, for fixing a miss offline."""
    ordered = sorted(rows, key=lambda row: -float(row[1]))[:limit]
    lines = [
        f"top {len(ordered)} kernels by device time (of {len(rows)} distinct)",
        f"{'total ms':>10} {'launches':>8}  {'category':<18} name",
    ]
    for row in ordered:
        name, time_us = row[0], float(row[1])
        count = int(row[2]) if len(row) > 2 else 1
        lines.append(f"{time_us / 1000.0:>10.1f} {count:>8d}  {h3_profile_category(name):<18} {name}")
    return "\n".join(lines)


def device_busy_us(intervals) -> float:
    """Length of the union of ``(start_us, end_us)`` intervals.

    Kernels and host<->device copies run concurrently on different streams, so their
    durations sum to more than the time the device was busy; the union is the busy time."""
    busy = 0.0
    current_start = current_end = None
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                busy += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    if current_end is not None:
        busy += current_end - current_start
    return busy


H3_PROFILE_SCOPE_PREFIX = "h3."
H3_PROFILE_ROLE_PREFIX = "h3.forward."
H3_PROFILE_NO_SCOPE = "(no h3 scope)"
H3_PROFILE_PHASE_RECOMPUTE = "recompute"
H3_PROFILE_PHASE_BACKWARD = "backward"
H3_PROFILE_PHASE_BACKWARD_UNATTRIBUTED = "backward (unattributed)"
H3_PROFILE_PHASE_FORWARD_UNROLED = "forward (unroled)"
H3_PROFILE_PHASE_OTHER = "other"
H3_PROFILE_STEP_PREFIX = "ProfilerStep#"
H3_PROFILE_OPTIMIZER_PREFIX = "Optimizer.step"
_H3_PROFILE_BACKWARD_SCOPE = 1  # torch RecordScope.BACKWARD_FUNCTION, ``FunctionEvent.scope``


def _enclosing_annotations(annotations, intervals):
    """For every ``(start, end)`` in ``intervals`` the indices of the ``annotations`` enclosing it, outermost first.

    ``annotations`` are ``(start, end, name)``. One sweep in start order keeps a stack of the labels
    still open; a label is pushed once and popped once, and every interval filters the open stack
    for the labels that also end after it (a label that only partially overlaps an interval is not
    an enclosing one). Cost: the sort, O((n + m) log(n + m)) for n labels and m intervals, plus
    O(m * depth) for the per-interval filter, where depth is the label nesting (about six here)."""
    items = [(start, 0, -end, index) for index, (start, end, _) in enumerate(annotations)]
    items.extend((start, 1, -end, index) for index, (start, end) in enumerate(intervals))
    items.sort()
    stack: list[int] = []
    result: list[list[int]] = [[] for _ in intervals]
    for start, kind, negative_end, index in items:
        while stack and annotations[stack[-1]][1] <= start:
            stack.pop()
        if kind == 0:
            stack.append(index)
        else:
            end = -negative_end
            result[index] = [candidate for candidate in stack if annotations[candidate][1] >= end]
    return result


def _scope_and_role(annotations, enclosing) -> tuple[str | None, str | None]:
    """The innermost non-role ``h3.*`` label and the outermost ``h3.forward.<role>`` label."""
    scope = None
    role = None
    for index in enclosing:
        name = annotations[index][2]
        if name.startswith(H3_PROFILE_ROLE_PREFIX):
            if role is None:
                role = name[len(H3_PROFILE_ROLE_PREFIX) :]
        else:
            scope = name
    return scope, role


def _backward_node(event):
    """The autograd node (``scope == BACKWARD_FUNCTION``) a CPU op runs under, or ``None``."""
    seen = 0
    while event is not None and seen < 64:
        if getattr(event, "scope", 0) == _H3_PROFILE_BACKWARD_SCOPE:
            return event
        event = getattr(event, "cpu_parent", None)
        seen += 1
    return None


def attribute_h3_profile_scopes(events, device_type: str) -> dict:
    """Attribute every leaf kernel to the innermost ``h3.*`` label enclosing it, per phase.

    ``record_function`` labels show up on the device timeline as user annotations spanning the
    kernels launched under them; a kernel belongs to the innermost such label. The enclosing
    ``h3.forward.<role>`` label names the phase (``student``, ``teacher``, ``rollout``, ...).

    Everything outside a role is placed by the *backward window* of its step: from the end of the
    step's last role label to the start of its optimizer label (``Optimizer.step#...``) or, failing
    that, the end of the ``ProfilerStep#`` label (the whole trace when there is none). A labelled
    kernel inside the window ran in a checkpoint recompute, which re-enters the labels from inside
    backward; a labelled kernel outside every role and every window is ``forward (unroled)``. Both
    are time-window heuristics.

    An unlabelled kernel is a backward kernel when its launching CPU op runs under an autograd
    node, and is then charged to the label of the forward op that node differentiates (the node's
    ``(sequence_nr, fwd_thread)`` names that op). The launch link is the kernel's
    ``linked_correlation_id`` (torch >= 2.10) or the CPU runtime record sharing the kernel's
    correlation ``id`` (whose parent is the op); when no kernel of the trace links at all, the
    per-scope backward attribution is reported unavailable and the unlabelled kernels of every
    backward window go to one ``backward (unattributed)`` row instead of being guessed. What is
    left -- loss, optimizer, data movement -- is ``(no h3 scope)`` / ``other``.

    Returns ``{"rows": {(scope, phase): (time_us, launches, {category: time_us})}, "forwards":
    {role: count}, "labels": {(scope, phase): count}, "linked": bool}``; ``labels`` counts the
    annotations themselves, so ``launches / labels`` is the launch count per labelled region.
    Without a device the CPU ops stand in for kernels and the CPU-side labels for the device ones."""
    cuda_type = getattr(torch.autograd.DeviceType, "CUDA", None)
    on_device = device_type == "cuda"
    annotations: list[tuple[float, float, str]] = []
    cpu_annotations: dict[int, list[tuple[float, float, str]]] = {}
    step_spans: list[tuple[float, float]] = []
    optimizer_starts: list[float] = []
    kernels = []
    cpu_events: dict[int, object] = {}
    forward_ops: dict[tuple[int, int], object] = {}
    forwards: dict[str, int] = {}
    for event in events:
        name = str(event.name)
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            continue
        span = (float(time_range.start), float(time_range.end))
        is_device = getattr(event, "device_type", None) == cuda_type
        if getattr(event, "is_user_annotation", False) or name.startswith(H3_PROFILE_STEP_PREFIX):
            if is_device != on_device:
                if not is_device and name.startswith(H3_PROFILE_SCOPE_PREFIX):
                    cpu_annotations.setdefault(int(getattr(event, "thread", 0) or 0), []).append((*span, name))
                continue
            if name.startswith(H3_PROFILE_STEP_PREFIX):
                step_spans.append(span)
            elif name.startswith(H3_PROFILE_OPTIMIZER_PREFIX):
                optimizer_starts.append(span[0])
            elif name.startswith(H3_PROFILE_SCOPE_PREFIX):
                annotations.append((*span, name))
                if name.startswith(H3_PROFILE_ROLE_PREFIX):
                    role = name[len(H3_PROFILE_ROLE_PREFIX) :]
                    forwards[role] = forwards.get(role, 0) + 1
                if not is_device:
                    cpu_annotations.setdefault(int(getattr(event, "thread", 0) or 0), []).append((*span, name))
            continue
        if not is_device:
            identifier = getattr(event, "id", None)
            if identifier is not None:
                cpu_events[int(identifier)] = event
        if getattr(event, "is_async", False):
            continue
        if not is_device:
            sequence_nr = int(getattr(event, "sequence_nr", -1) or -1)
            if sequence_nr >= 0 and getattr(event, "scope", 0) != _H3_PROFILE_BACKWARD_SCOPE:
                forward_ops.setdefault((sequence_nr, int(getattr(event, "thread", 0) or 0)), event)
        if on_device:
            if not is_device:
                continue
            time_us = span[1] - span[0]
        else:
            time_us = float(event.self_cpu_time_total)
        kernels.append((span, name, time_us, event))

    windows = _backward_windows(
        step_spans, [span for *span, name in annotations if name.startswith(H3_PROFILE_ROLE_PREFIX)], optimizer_starts
    )
    window_starts = [start for start, _ in windows]

    def in_backward_window(time_us: float) -> bool:
        index = bisect.bisect_right(window_starts, time_us) - 1
        return index >= 0 and time_us < windows[index][1]

    # Phase of every label: the role enclosing it, else recompute inside a backward window (a label
    # re-entered from inside backward), else forward (unroled). Counted per (scope, phase).
    labels: dict[tuple[str, str], int] = {}
    label_spans = [(start, end) for start, end, _ in annotations]
    for index, enclosed in enumerate(_enclosing_annotations(annotations, label_spans)):
        start, _, name = annotations[index]
        if name.startswith(H3_PROFILE_ROLE_PREFIX):
            continue
        _, role = _scope_and_role(annotations, [candidate for candidate in enclosed if candidate != index])
        if role is not None:
            phase = role
        else:
            phase = H3_PROFILE_PHASE_RECOMPUTE if in_backward_window(start) else H3_PROFILE_PHASE_FORWARD_UNROLED
        labels[(name, phase)] = labels.get((name, phase), 0) + 1

    enclosing = _enclosing_annotations(annotations, [span for span, *_ in kernels])
    # Forward ops referenced by backward nodes, labelled by their CPU-side annotations per thread.
    referenced: dict[int, list] = {}
    pending = []
    linked = not on_device and bool(kernels)
    for (span, name, time_us, event), enclosed in zip(kernels, enclosing):
        scope, role = _scope_and_role(annotations, enclosed)
        launcher = _kernel_launcher(event, cpu_events) if on_device else event
        linked = linked or launcher is not None
        node = _backward_node(launcher) if role is None and launcher is not None else None
        forward_op = None
        if node is not None and scope is None:
            key = (int(getattr(node, "sequence_nr", -1) or -1), int(getattr(node, "fwd_thread", 0) or 0))
            forward_op = forward_ops.get(key)
            if forward_op is not None:
                referenced.setdefault(int(getattr(forward_op, "thread", 0) or 0), []).append(forward_op)
        pending.append(
            (name, time_us, scope, role, launcher is not None, node is not None, in_backward_window(span[0]), forward_op)
        )
    forward_scopes: dict[int, str | None] = {}
    for thread, ops in referenced.items():
        thread_annotations = cpu_annotations.get(thread, [])
        spans = [(float(op.time_range.start), float(op.time_range.end)) for op in ops]
        for op, enclosed in zip(ops, _enclosing_annotations(thread_annotations, spans)):
            forward_scopes[id(op)] = _scope_and_role(thread_annotations, enclosed)[0]

    rows: dict[tuple[str, str], list] = {}
    for name, time_us, scope, role, has_launcher, under_node, in_window, forward_op in pending:
        if role is not None:
            phase = role
        elif scope is not None:
            phase = H3_PROFILE_PHASE_RECOMPUTE if (under_node or in_window) else H3_PROFILE_PHASE_FORWARD_UNROLED
        elif under_node:
            phase = H3_PROFILE_PHASE_BACKWARD
            if forward_op is not None:
                scope = forward_scopes.get(id(forward_op))
        elif in_window and not has_launcher:
            phase = H3_PROFILE_PHASE_BACKWARD_UNATTRIBUTED
        else:
            phase = H3_PROFILE_PHASE_OTHER
        entry = rows.setdefault((scope or H3_PROFILE_NO_SCOPE, phase), [0.0, 0, {}])
        entry[0] += time_us
        entry[1] += 1
        category = h3_profile_category(name)
        entry[2][category] = entry[2].get(category, 0.0) + time_us
    return {
        "rows": {key: (time_us, launches, categories) for key, (time_us, launches, categories) in rows.items()},
        "forwards": forwards,
        "labels": labels,
        "linked": linked,
        "optimizer_labelled": bool(optimizer_starts),
    }


def _kernel_launcher(kernel, cpu_events: dict[int, object]):
    """The CPU-side event a device kernel was launched from, or ``None`` when the trace has no link.

    torch >= 2.10 stamps ``linked_correlation_id`` (the launching op's ``id``) on the kernel event;
    older releases only give the kernel the CUPTI correlation ``id`` it shares with the CPU runtime
    record (``cudaLaunchKernel``), which the profiler nests under the op, so that record's parent
    chain leads to the op. Either way the caller walks ``cpu_parent`` upwards from here."""
    linked = getattr(kernel, "linked_correlation_id", 0) or 0
    if linked:
        launcher = cpu_events.get(int(linked))
        if launcher is not None:
            return launcher
    identifier = getattr(kernel, "id", None)
    if identifier is None:
        return None
    runtime = cpu_events.get(int(identifier))
    if runtime is None or runtime is kernel:
        return None
    # A CPU event sharing the kernel's correlation id is its CUDA runtime launch
    # record only if it looks like one and is nested under a launching op; an id
    # that merely collides with an unrelated CPU op must not name a launcher.
    if not _LAUNCH_RECORD.search(str(getattr(runtime, "name", ""))):
        return None
    if getattr(runtime, "cpu_parent", None) is None:
        return None
    return runtime


_LAUNCH_RECORD = re.compile(r"LaunchKernel|LaunchCooperativeKernel|cuLaunch|launch_kernel", re.IGNORECASE)


def _backward_windows(step_spans, role_spans, optimizer_starts) -> list[tuple[float, float]]:
    """Per step, the window after its last forward role and before its optimizer label (else its end).

    Steps are the ``ProfilerStep#`` spans, or one span over everything when the trace has none; a
    step with no role label has no backward window."""
    if not role_spans:
        return []
    if not step_spans:
        step_spans = [(float("-inf"), float("inf"))]
    role_ends = sorted(end for _, end in role_spans)
    optimizer_starts = sorted(optimizer_starts)
    windows = []
    for step_start, step_end in sorted(step_spans):
        index = bisect.bisect_right(role_ends, step_end) - 1
        if index < 0 or role_ends[index] < step_start:
            continue
        start = role_ends[index]
        optimizer_index = bisect.bisect_right(optimizer_starts, start)
        end = step_end
        if optimizer_index < len(optimizer_starts) and optimizer_starts[optimizer_index] < step_end:
            end = optimizer_starts[optimizer_index]
        if end > start:
            windows.append((start, end))
    return windows


def format_h3_profile_scopes(scopes: dict, wall_us: float, steps: int) -> str:
    """Render the per-scope table of :func:`attribute_h3_profile_scopes`."""
    steps = max(int(steps), 1)
    rows = sorted(scopes["rows"].items(), key=lambda item: -item[1][0])
    forwards = ", ".join(f"{role} {count}" for role, count in sorted(scopes["forwards"].items())) or "none labelled"
    header = (
        f"{'scope':<16} {'phase':<10} {'total ms':>10} {'ms/step':>9} {'share':>7} {'launches':>9} {'labels':>7} "
        f"{'gemm ms':>9} {'attn ms':>9} {'elem ms':>9} {'swap ms':>8}"
    )
    if scopes.get("linked", True):
        backward_note = "backward = charged to the label of the forward op its autograd node differentiates"
    else:
        backward_note = (
            "per-scope backward attribution is UNAVAILABLE on this torch (kernel events carry no launch link), so "
            "unlabelled kernels of each backward window are one 'backward (unattributed)' row"
        )
        if not scopes.get("optimizer_labelled", True):
            backward_note += (
                "; no Optimizer.step label was found, so each window runs to the end of its step and that row "
                "INCLUDES the optimizer's kernels"
            )
    lines = [
        (
            "device time by innermost h3.* scope and phase (a kernel belongs to the innermost label enclosing it on the "
            "device timeline; the phase is the enclosing h3.forward.<role> label; recompute detection is time-window "
            "based: a label re-entered after a step's last forward role and before its optimizer label is a recompute, "
            "one outside every role and window is 'forward (unroled)'; "
            f"{backward_note}; labels = entries of the label in that phase, so launches/labels is the launch count per "
            "labelled region)"
        ),
        f"forwards in the window: {forwards}",
        header,
    ]
    for (scope, phase), (time_us, launches, categories) in rows:
        share = time_us / wall_us if wall_us > 0 else 0.0
        label_count = scopes["labels"].get((scope, phase), 0)
        lines.append(
            f"{scope:<16} {phase:<10} {time_us / 1000.0:>10.1f} {time_us / 1000.0 / steps:>9.1f} {share:>6.1%} {launches:>9d} "
            f"{label_count:>7d} {categories.get('gemm', 0.0) / 1000.0:>9.1f} {categories.get('attention', 0.0) / 1000.0:>9.1f} "
            f"{categories.get(H3_PROFILE_OTHER, 0.0) / 1000.0:>9.1f} {categories.get('swap', 0.0) / 1000.0:>8.1f}"
        )
    return "\n".join(lines)


def format_h3_profile_table(
    categories: dict[str, tuple[float, int]], wall_us: float, steps: int, busy_us: float | None = None, note: str = ""
) -> str:
    """Render the per-category table.

    ``busy_us`` is the union of the kernel intervals (``device_busy_us``); idle is the wall time
    outside it. Category shares are of the wall time and can sum to more than the busy share,
    because kernels overlap across streams. Without ``busy_us`` the plain kernel-time sum stands
    in, which overstates busy time wherever streams overlap."""
    kernel_us = sum(time_us for time_us, _ in categories.values())
    if busy_us is None:
        busy_us = kernel_us
    rows = list(categories.items()) + [(H3_PROFILE_IDLE, (max(wall_us - busy_us, 0.0), 0))]
    steps = max(int(steps), 1)
    header = (
        f"profile of {steps} optimizer step(s){note}, wall {wall_us / 1000.0:.1f} ms ({wall_us / 1000.0 / steps:.1f} ms/step), "
        f"device busy {busy_us / 1000.0:.1f} ms as the union of kernel intervals, kernel durations sum to "
        f"{kernel_us / 1000.0:.1f} ms (the excess is cross-stream overlap)"
    )
    lines = [header, f"{'category':<18} {'total ms':>10} {'ms/step':>10} {'share':>7} {'kernels':>8}"]
    for name, (time_us, count) in rows:
        share = time_us / wall_us if wall_us > 0 else 0.0
        lines.append(f"{name:<18} {time_us / 1000.0:>10.1f} {time_us / 1000.0 / steps:>10.1f} {share:>6.1%} {count:>8d}")
    return "\n".join(lines)


class H3StepProfiler:
    """Profile ``active_steps`` optimizer steps with ``torch.profiler`` and report one table.

    ``step()`` is called once per optimizer step. The first ``H3_PROFILE_WAIT_STEPS`` steps are
    skipped (the first one carries lazy initialisation and compilation), the next
    ``H3_PROFILE_WARMUP_STEPS`` run the profiler with the trace discarded, and the following
    ``active_steps`` are recorded. The wall time of the recorded window is measured between
    device synchronisations at its two ends, and the idle row is the wall time outside the
    union of the kernel intervals. Once the table has been written the profiler is stopped and
    ``finished`` is set; ``abort()`` writes whatever the window holds when training ends first."""

    def __init__(self, active_steps: int, device: torch.device, output_path: Path) -> None:
        if active_steps <= 0:
            raise ValueError("H3StepProfiler needs a positive number of steps")
        self.active_steps = int(active_steps)
        self.device = device
        self.output_path = Path(output_path)
        self.finished = False
        self.table: str | None = None
        self._steps = 0
        self._window_start: float | None = None
        self._window_end: float | None = None
        self._rows: list[tuple[str, float, int]] | None = None
        self._scopes: dict | None = None
        self._busy_us: float = 0.0
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=H3_PROFILE_WAIT_STEPS, warmup=H3_PROFILE_WARMUP_STEPS, active=self.active_steps, repeat=1
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=False,
        )

    @property
    def _first_active_step(self) -> int:
        return H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS

    def start(self) -> None:
        self._profiler.start()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def active_steps_done(self) -> int:
        return min(max(self._steps - self._first_active_step, 0), self.active_steps)

    def _on_trace_ready(self, profiler) -> None:
        events = profiler.events()
        self._rows, self._busy_us = collect_h3_profile_rows(events, self.device.type)
        self._scopes = attribute_h3_profile_scopes(events, self.device.type)

    def step(self) -> None:
        self._steps += 1
        if self._steps == self._first_active_step + self.active_steps:
            self._synchronize()
            self._window_end = time.perf_counter()
        self._profiler.step()
        if self._steps == self._first_active_step:
            self._synchronize()
            self._window_start = time.perf_counter()
        if self._rows is None:
            return
        self._finish("")

    def _wall_us(self) -> float:
        if self._window_start is None or self._window_end is None:
            return 0.0
        return max(self._window_end - self._window_start, 0.0) * 1e6

    def _finish(self, note: str) -> None:
        steps = self.active_steps_done if note else self.active_steps
        self.table = format_h3_profile_table(categorize_h3_profile_rows(self._rows), self._wall_us(), steps, self._busy_us, note)
        self._profiler.stop()
        sections = [self.table]
        if self._scopes is not None:
            sections.append(format_h3_profile_scopes(self._scopes, self._wall_us(), steps))
        sections.append(format_h3_profile_top_kernels(self._rows))
        self._write("\n\n".join(sections))
        logger.info("--h3_profile_steps table (also written to %s):\n%s", self.output_path, self.table)

    def _write(self, text: str) -> None:
        self.finished = True
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(text + "\n", encoding="utf-8")

    def abort(self) -> None:
        """Training ended before the window completed: report what the window holds."""
        if self.finished:
            return
        if self._window_start is not None and self._window_end is None:
            self._synchronize()
            self._window_end = time.perf_counter()
        # Stopping inside the active phase delivers the partial trace to _on_trace_ready.
        self._profiler.stop()
        if self._rows is not None and self.active_steps_done > 0:
            self._finish(f" of the {self.active_steps} requested, INCOMPLETE")
            return
        message = (
            f"--h3_profile_steps {self.active_steps}: the profiling window never opened; training ended after "
            f"{self._steps} optimizer step(s) and the window starts after {self._first_active_step}"
        )
        self._write(message)
        logger.warning(message)


def _parse_block_sparse_block_shape(spec: str | None) -> tuple[int, int, int] | None:
    """Parse a frames,height,width tile whose product is the sparse block size."""
    if spec is None or not spec.strip():
        return None
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 3:
        raise ValueError(f"--h3_block_sparse_block_shape must be frames,height,width, got {spec!r}")
    try:
        extents = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"--h3_block_sparse_block_shape must be three integers, got {spec!r}") from error
    if any(extent < 1 for extent in extents):
        raise ValueError(f"--h3_block_sparse_block_shape extents must be positive, got {spec!r}")
    if extents[0] * extents[1] * extents[2] != DEFAULT_BLOCK:
        raise ValueError(f"--h3_block_sparse_block_shape must multiply to the block size ({DEFAULT_BLOCK}), got {spec!r}")
    return extents


def _parse_h3_block_ranges(spec: str) -> tuple[int, ...]:
    """Parse a compact block selection such as ``0-7,16,24-31``."""
    selected: set[int] = set()
    for raw_piece in spec.split(","):
        piece = raw_piece.strip()
        if not piece:
            raise ValueError("H3 LoRA target contains an empty block selection")
        if "-" in piece:
            endpoints = piece.split("-", 1)
            if len(endpoints) != 2 or not all(endpoint.isdigit() for endpoint in endpoints):
                raise ValueError(f"invalid H3 LoRA block range {piece!r}")
            first, last = (int(endpoint) for endpoint in endpoints)
            if first > last:
                raise ValueError(f"H3 LoRA block range {piece!r} is descending")
            selected.update(range(first, last + 1))
        elif piece.isdigit():
            selected.add(int(piece))
        else:
            raise ValueError(f"invalid H3 LoRA block entry {piece!r}")
    if not selected:
        raise ValueError("H3 LoRA target must select at least one block")
    if max(selected) >= _H3_TRANSFORMER_BLOCKS:
        raise ValueError(f"H3 LoRA block indices must be between 0 and {_H3_TRANSFORMER_BLOCKS - 1}")
    return tuple(sorted(selected))


def _parse_h3_lora_targets(spec: str | None) -> dict[str, tuple[int, ...] | None]:
    """Parse ``attention:0-13;mlp:3-5;audio`` into named target groups."""
    if spec is None:
        return {}
    selected: dict[str, tuple[int, ...] | None] = {}
    for raw_entry in spec.split(";"):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError("--h3_lora_targets contains an empty target")
        name, separator, ranges = entry.partition(":")
        name = name.strip()
        if name not in _H3_LORA_TARGETS:
            raise ValueError(f"unknown H3 LoRA target {name!r}; choose from {', '.join(_H3_LORA_TARGETS)}")
        if name in selected:
            raise ValueError(f"--h3_lora_targets repeats {name!r}")
        if separator and name not in {"attention", "mlp"}:
            raise ValueError(f"H3 LoRA target {name!r} does not have numbered main blocks")
        if separator and not ranges.strip():
            raise ValueError(f"H3 LoRA target {name!r} has an empty block range")
        selected[name] = _parse_h3_block_ranges(ranges.strip()) if separator else None
    return selected


def load_measured_variance_curve(path: str, weight_max: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-modality loss weights from the target dispersion a ``probe_g_curve`` report measured.

    The report holds, per sigma bucket and modality, ``g`` = ||mean over noise
    draws of the residual|| / ||target|| and ``raw`` = mean over draws of
    ||residual|| / ||target||. ``raw^2 - g^2`` is a dimensionless DISPERSION
    PROXY for the target at that noise level -- not its variance: ``raw`` is a
    mean of norms rather than a root mean square (Jensen makes the difference
    an underestimate), and the target-norm scaling is divided out. It orders the
    buckets by how noisy the one-draw target is there, which is all the
    weighting uses it for.

    Weighting samples by the inverse of that proxy is a HEURISTIC sigma
    reweighting: the per-sigma gradients estimate different conditional
    updates, so this changes the objective (noisy noise levels count for less)
    rather than reducing the variance of one fixed estimator. Weights are
    normalised to mean 1 over the buckets -- not over the sigma distribution
    actually sampled, so the effective step scale is only approximately
    preserved -- then capped at ``weight_max`` so a nearly noise-free bucket
    cannot dominate the step (a binding cap leaves the mean slightly below 1).
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = report.get("summary") if isinstance(report, dict) else None
    if not rows:
        raise ValueError(f"--h3_measured_variance_weighting: {path} has no 'summary' rows")
    curves: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for modality in ("video", "audio"):
        subset = sorted((r for r in rows if r.get("modality") == modality), key=lambda r: float(r["sigma"]))
        if not subset:
            continue
        sigmas = torch.tensor([float(r["sigma"]) for r in subset], dtype=torch.float32)
        variance = torch.tensor(
            [max(float(r["raw_mean"]) ** 2 - float(r["g_mean"]) ** 2, 1e-8) for r in subset], dtype=torch.float32
        )
        weights = 1.0 / variance
        weights = weights / weights.mean()
        # The cap is final: renormalising after it would lift the capped bucket
        # straight back over the line. A binding cap therefore lowers the mean
        # a little below 1, which is a slightly smaller effective step, never a
        # larger one.
        weights = weights.clamp_max(float(weight_max))
        curves[modality] = (sigmas, weights)
    if "video" not in curves:
        raise ValueError(f"--h3_measured_variance_weighting: {path} carries no video rows")
    return curves


def interpolate_curve(sigmas: torch.Tensor, weights: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear lookup of ``weights`` at ``query``, flat beyond the ends."""
    grid = sigmas.to(device=query.device, dtype=torch.float32)
    values = weights.to(device=query.device, dtype=torch.float32)
    q = query.to(dtype=torch.float32).clamp(min=float(grid[0]), max=float(grid[-1]))
    upper = torch.searchsorted(grid, q).clamp(1, grid.numel() - 1)
    lower = upper - 1
    span = (grid[upper] - grid[lower]).clamp_min(1e-8)
    t = (q - grid[lower]) / span
    return values[lower] + t * (values[upper] - values[lower])


def _anchor_probability(args) -> float:
    """--h3_guidance_null_anchor_probability with an absent value meaning 1, not a zero."""
    value = getattr(args, "h3_guidance_null_anchor_probability", None)
    return 1.0 if value is None else float(value)


def _apply_timestep_focus(base: torch.Tensor, low: float, high: float, probability: float) -> torch.Tensor:
    """Map one uniform draw to a uniform/background mixture without another RNG draw."""
    if probability <= 0.0:
        return base
    if probability >= 1.0:
        return low + (high - low) * base
    focused = low + (high - low) * (base / probability)
    background = (base - probability) / (1.0 - probability)
    return torch.where(base < probability, focused, background)


def _validate_dataset_loss_coverage(user_config: dict, *, video_weight: float, audio_weight: float) -> None:
    """Reject dataset rows that can never contribute to the configured objective."""
    general = user_config.get("general", {})
    for index, dataset in enumerate(user_config.get("datasets", [])):
        is_image = bool(dataset.get("image_directory") or dataset.get("image_jsonl_file") or dataset.get("target_image_directory"))
        is_video = bool(dataset.get("video_directory") or dataset.get("video_jsonl_file") or dataset.get("target_video_directory"))
        if is_image:
            modalities = tuple(dataset.get("target_modalities", ("image",)))
            mode = "av" if modalities == ("image", "audio") else "video"
        elif is_video:
            modalities = tuple(dataset.get("target_modalities", ("video", "audio")))
            mode = "video" if modalities == ("video",) else "av"
        elif dataset.get("audio_directory") or dataset.get("audio_jsonl_file") or dataset.get("target_audio_directory"):
            mode = "audio"
        else:
            mode = dataset.get("h3_target_mode", general.get("h3_target_mode", "av"))
        active = (mode in {"av", "video"} and video_weight > 0) or (mode in {"av", "audio"} and audio_weight > 0)
        if not active:
            raise ValueError(f"H3 dataset {index + 1} has target mode {mode!r}, but its configured modality loss weight is zero")


class _H3DecoderBundle(torch.nn.Module):
    def __init__(self, video_decoder: torch.nn.Module, audio_decoder: torch.nn.Module) -> None:
        super().__init__()
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder


class _IndexedValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_index = self.indices[index]
        return dataset_index, self.dataset[dataset_index]


def _parse_keyframe_anchors(spec: str) -> tuple[int | str, ...]:
    """Parse a keyframe anchor spec into 'first'/'last' markers and frame indices."""
    if not spec:
        return ()
    anchors: list[int | str] = []
    for piece in spec.split(","):
        token = piece.strip()
        if token in ("first", "last"):
            anchors.append(token)
        elif token.lstrip("-").isdigit():
            anchors.append(int(token))
        else:
            raise ValueError(f"H3 keyframe anchor {token!r} must be 'first', 'last', or a latent frame index")
    return tuple(anchors)


# The cheapest configuration that was not measured worse than a longer rollout: every
# supervised state adds a trainable forward and a teacher forward, so the window is what
# the objective is charged for.
_ROLLOUT_PROBABILITY_DEFAULT = 0.5
_ROLLOUT_STEPS_DEFAULT = 2
_ROLLOUT_WINDOW_DEFAULT = 1

# The arguments every H3 training backend has taken positionally since the first
# one. A described forward is a dict, and these are the entries that go back on
# the wire in order rather than by name.
_PREDICT_POSITIONAL_KEYS = frozenset(("batch", "video_hidden_states", "audio_hidden_states", "video_timestep", "audio_timestep"))


@contextmanager
def _forked_frozen_build(fork_devices: list, set_enabled):
    """Build one frozen arm's packed sequence without disturbing the others.

    Entered once per fused forward, around the stage that draws: the conditioning
    rows a Ref2VA or keyframe presentation jitters come out of the global stream,
    and a frozen arm must not consume the draws the trainable arm would have made.
    The adapter is off here too, because a token refiner carrying LoRA is part of
    the build rather than of the block loop.
    """
    with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
        set_enabled(False)
        try:
            yield
        finally:
            set_enabled(True)


def _parse_guide_specs(spec: str) -> tuple[tuple[int, int, int], ...]:
    """Parse ``START:VIDEO_LATENTS:AUDIO_LATENTS`` guide recipes."""
    if not spec:
        return ()
    guides = []
    for piece in spec.split(";"):
        fields = [field.strip() for field in piece.split(":")]
        if len(fields) != 3 or any(not field.lstrip("-").isdigit() for field in fields):
            raise ValueError(f"H3 guide {piece!r} must be START:VIDEO_LATENTS:AUDIO_LATENTS; separate multiple guides with ';'")
        start, video_latents, audio_latents = (int(field) for field in fields)
        if video_latents < 0 or audio_latents < 0 or not (video_latents or audio_latents):
            raise ValueError("H3 guide stream lengths must be non-negative and at least one must be non-zero")
        guides.append((start, video_latents, audio_latents))
    return tuple(guides)


def _parse_guidance_scale_range(spec: str | None) -> tuple[float, float] | None:
    """Parse ``--h3_guidance_scale_range LOWER,UPPER`` into a validated pair."""
    if not spec:
        return None
    pieces = [piece.strip() for piece in str(spec).split(",")]
    if len(pieces) != 2:
        raise ValueError(f"--h3_guidance_scale_range must be 'LOWER,UPPER', got {spec!r}")
    try:
        lower, upper = (float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"--h3_guidance_scale_range must be 'LOWER,UPPER', got {spec!r}") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("--h3_guidance_scale_range bounds must be finite")
    if lower <= 1.0:
        raise ValueError("--h3_guidance_scale_range lower bound must be greater than 1")
    if lower > upper:
        raise ValueError("--h3_guidance_scale_range lower bound must not exceed its upper bound")
    return (lower, upper)


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    @staticmethod
    def _build_audio_only_spatial_tokens(audio_latents: torch.Tensor) -> torch.Tensor:
        """Build H3's audio-only spatial placeholders, one token per latent frame."""
        if audio_latents.ndim != 4:
            raise ValueError("H3 audio-only spatial tokens require [B, 2, C, T] audio latents")
        pixel_frames = max(2, round(int(audio_latents.shape[-1]) / AUDIO_LATENT_FPS * VIDEO_FPS))
        latent_frames = temporal_shape(align_frame_count(pixel_frames)).video_latent_frames
        patch_t, patch_h, patch_w = VIDEO_DIT_PATCH_SIZE
        if latent_frames % patch_t:
            latent_frames += patch_t - latent_frames % patch_t
        return torch.zeros(
            audio_latents.shape[0],
            VIDEO_LATENT_CHANNELS,
            latent_frames,
            patch_h,
            patch_w,
            device=audio_latents.device,
            dtype=audio_latents.dtype,
        )

    @staticmethod
    def _sparse_branch_active(accelerator: Accelerator, probability: float, generator: torch.Generator | None = None) -> bool:
        """Draw one auxiliary-branch decision shared by every distributed rank."""
        if probability >= 1.0:
            return True
        device = accelerator.device
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        # Draw on CPU before any replayed model branch. Every rank advances its
        # own seeded CPU stream once; rank zero's decision is then authoritative.
        # This keeps the CUDA replay untouched without restoring the Bernoulli
        # generator to the same position after every call.
        active = (torch.rand((), device="cpu", generator=generator) < probability).to(device=device)
        if distributed:
            torch.distributed.broadcast(active, src=0)
        return bool(active.item())

    @staticmethod
    def _sparse_branch_choice(accelerator: Accelerator, weights, generator: torch.Generator | None = None) -> int:
        """Draw one categorical branch index shared by every distributed rank.

        The same contract as ``_sparse_branch_active`` generalized past two
        outcomes: one CPU draw off the caller's stream, bucketed by the
        cumulative weights, then broadcast so every rank builds the same
        conditioning. An index equal to ``len(weights)`` means the residual
        ``1 - sum(weights)`` outcome was drawn.
        """
        draw = float(torch.rand((), device="cpu", generator=generator))
        index = len(weights)
        cumulative = 0.0
        for position, weight in enumerate(weights):
            cumulative += float(weight)
            if draw < cumulative:
                index = position
                break
        selected = torch.tensor(index, device=accelerator.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(selected, src=0)
        return int(selected.item())

    def _base_preservation_active(self, accelerator: Accelerator, probability: float) -> bool:
        """Draw one preservation decision shared by every distributed rank.

        On its own stream, for the reason the guidance branch already is: drawn
        off the ambient CPU stream it would advance it once per step before the
        rollout generators are lazily seeded FROM that stream, so enabling sparse
        preservation silently reseeded them and changed which steps a seeded
        rollout run selected.
        """
        if probability >= 1.0:
            return True
        generator = self._preservation_probability_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._preservation_probability_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _null_anchor_probability_active(self, accelerator: Accelerator, probability: float) -> bool:
        """The same, for the null anchor's own sparse draw (--h3_guidance_null_anchor_probability)."""
        if probability >= 1.0:
            return True
        generator = getattr(self, "_null_anchor_probability_generator", None)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._null_anchor_probability_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _dop_probability_active(self, accelerator: Accelerator, probability: float) -> bool:
        """The same, for differential output preservation's own sparse draw."""
        if probability >= 1.0:
            return True
        generator = self._dop_probability_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._dop_probability_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _guidance_distillation_active(self, accelerator: Accelerator, probability: float) -> bool:
        """Draw one guidance-distillation decision shared by every distributed rank."""
        if probability >= 1.0:
            return True
        generator = self._guidance_probability_generator
        if generator is None:
            # A dedicated stream keeps the two sparse objectives independent:
            # enabling guidance sparsity must not shift the global CPU draws the
            # preservation branch, caption dropout, and the jitters consume. The
            # seed still comes from the global stream, so a seeded run remains
            # reproducible.
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._guidance_probability_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _draw_guidance_scale(self, accelerator: Accelerator, batch_size: int) -> torch.Tensor:
        """Draw one guidance scale per micro-batch sample, shared by every rank.

        A fifth dedicated stream, independent of the guidance-sparsity, recipe,
        preservation and control-dropout ones and of the global CPU stream that
        caption dropout, the observed-modality draw and the jitters consume:
        enabling ``--h3_guidance_scale_range`` must not shift any of them, so a
        run that only adds the range keeps every other branch decision it had
        with a fixed point scale. Only the one-time seed comes from the global
        stream, so a seeded run stays reproducible.

        The draw is made on CPU and rank zero's vector is then authoritative,
        exactly as in :meth:`_sparse_branch_active`: every rank must correct the
        same guided field, or the gradients being reduced belong to different
        objectives.
        """
        if self._guidance_scale_range is None:
            raise RuntimeError("H3 guidance scale range was not configured")
        lower, upper = self._guidance_scale_range
        generator = self._guidance_scale_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._guidance_scale_generator = generator
        draw = torch.rand(batch_size, device="cpu", dtype=torch.float32, generator=generator)
        scale = (lower + (upper - lower) * draw).to(device=accelerator.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(scale, src=0)
        return scale

    def _qwen_control_dropout_active(self, accelerator: Accelerator, probability: float) -> bool:
        """Draw one EXPERIMENTAL Qwen-control dropout decision, shared by every rank.

        A fourth dedicated stream, independent of the guidance, recipe and
        preservation ones and of the global CPU stream that caption dropout, the
        observed-modality draw and the jitters consume: enabling control dropout
        must not shift any of them. Only the one-time seed comes from the global
        stream, so a seeded run stays reproducible, and a rate of 0 never reaches
        this method, so it builds no generator and draws nothing.
        """
        if probability >= 1.0:
            return True
        generator = self._qwen_control_dropout_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._qwen_control_dropout_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _rollout_supervision_active(self, accelerator: Accelerator, probability: float) -> bool:
        """Draw one rollout-supervision decision shared by every distributed rank.

        A dedicated stream, independent of the guidance-sparsity, scale, recipe,
        preservation and control-dropout ones and of the global CPU stream that
        caption dropout, the observed-modality draw and the jitters consume:
        enabling rollout supervision must not shift any of them, so a run that
        only adds the flag keeps every other branch decision it had. Only the
        one-time seed comes from the global stream, so a seeded run stays
        reproducible.

        The decision itself is broadcast because it changes which objective the
        step optimizes: two ranks reducing gradients of different objectives is
        not the sparse estimator, it is a silent mixture.
        """
        if probability >= 1.0:
            return True
        generator = self._rollout_probability_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._rollout_probability_generator = generator
        return self._sparse_branch_active(accelerator, probability, generator)

    def _draw_rollout_stop_sigma(self, args: argparse.Namespace, video_shift: float = 1.0) -> float:
        """Draw the *unshifted* base sigma the rollout stops at.

        Uniform on the same base coordinate the data steps sample, and stratified
        over ``--num_timestep_buckets`` when bucketing is on, so the rollout visits
        the schedule with the same coverage the ordinary objective does rather
        than piling supervision at one noise level. The value is clamped away from
        both ends: sigma 1 is the noise state the rollout starts from and would
        make the rollout empty, and sigma 0 is a clean latent no sampler
        evaluates.

        ``--h3_rollout_stop_shifted`` draws that uniform on the *shifted* video
        coordinate instead and inverts the shift to get the base stop. The base
        grid is the wrong coordinate to be uniform on when the shift is 12: it maps
        almost the whole unit interval into shifted sigmas above 0.9, so a uniform
        base draw supervises only the noisiest band and never visits the mid band
        at all. Round 1 measured both halves of that -- the field was preserved
        where the rollout supervised and eroded at shifted sigma .6 -- while the
        D-OPSD preflight measured the teacher to be *strongest* exactly there
        (-50% relative velocity error at .6, -44% at .8). Audio is untouched: it
        keeps riding its own shift off whatever base value comes out, so the joint
        rollout stays synchronized.

        Its own dedicated stream, for the same reason as the decision above, and
        *not* broadcast: like the per-item data timestep, each rank supervises its
        own clip at its own noise level.
        """
        generator = self._rollout_noise_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._rollout_noise_generator = generator
        draw = float(torch.rand((), device="cpu", generator=generator))
        buckets = getattr(args, "num_timestep_buckets", None)
        if buckets is not None and buckets > 1:
            index = int(torch.randint(0, int(buckets), (), device="cpu", generator=generator))
            draw = (index + draw) / float(buckets)
        # The band lower bound lives on the same coordinate the draw is on, so
        # under --h3_rollout_stop_shifted it is a shifted sigma and otherwise a
        # base one; a data band capped with --h3_timestep_focus_max below it
        # gives every noise level exactly one master term.
        stop_min = float(getattr(args, "h3_rollout_stop_min", 0.0) or 0.0)
        if stop_min > 0:
            draw = stop_min + (1.0 - stop_min) * draw
        floor = ROLLOUT_SIGMA_FLOOR
        if getattr(args, "h3_rollout_stop_shifted", False) and video_shift != 1.0:
            # The draw *is* the shifted stop; clamp it there so the supervised
            # band is the one the flag names, then invert. A shift of 1 (an image
            # step) is the identity and skips the round trip entirely.
            shifted = min(max(draw, 2.0 * floor), 1.0 - floor)
            draw = float(unshift_sigma(torch.tensor([shifted], dtype=torch.float64), video_shift)[0])
        return min(max(draw, 2.0 * floor), 1.0 - floor)

    def _rollout_noise(self, reference: torch.Tensor) -> torch.Tensor:
        """The pure-noise state a rollout starts from, off the dedicated stream.

        Kept in fp32 whatever the DiT dtype is. The state is accumulated across
        every Euler step of the rollout, so a BF16 state would compound its ~3
        decimal digits over j additions; ``_predict`` casts to the compute dtype
        at each forward anyway, so the precision costs one buffer and nothing in
        the forward itself.
        """
        generator = self._rollout_noise_generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._rollout_noise_generator = generator
        return torch.randn(reference.shape, generator=generator, dtype=torch.float32).to(
            device=reference.device, dtype=torch.float32
        )

    def _draw_step_recipe(self, accelerator: Accelerator, args: argparse.Namespace) -> str | None:
        """Draw which conditioning recipe this step trains, shared by every rank.

        ``None`` means no mixing is configured and the step keeps whatever single
        recipe the flags select, bit-for-bit as before. Otherwise exactly one of
        ``mask``, ``extension`` or ``plain`` is drawn: masking and extension both
        claim the observed rows, so a step never carries both.
        """
        mask_configured, extension_configured = self._configured_recipes()
        mask_probability = float(args.h3_mask_probability) if mask_configured else 1.0
        extension_probability = float(args.h3_extension_probability) if extension_configured else 1.0
        if mask_probability >= 1.0 and extension_probability >= 1.0:
            return None
        generator = self._recipe_probability_generator
        if generator is None:
            # A third dedicated stream, independent of the guidance and
            # preservation ones and of the global CPU stream that caption
            # dropout, the observed-modality draw and the jitters consume:
            # enabling recipe mixing must not shift any of them. Only the
            # one-time seed comes from the global stream, so a seeded run stays
            # reproducible.
            generator = torch.Generator()
            generator.manual_seed(int(torch.randint(0, 1 << 62, (), device="cpu").item()))
            self._recipe_probability_generator = generator
        weights = (
            mask_probability if mask_configured else 0.0,
            extension_probability if extension_configured else 0.0,
        )
        return ("mask", "extension", "plain")[self._sparse_branch_choice(accelerator, weights, generator)]

    def _configured_recipes(self) -> tuple[bool, bool]:
        """Report whether masking and extension are configured for this run."""
        return (
            self._mask_mode != "off" or self._mask_audio,
            bool(self._extension_video_frames or self._extension_audio_latents),
        )

    @property
    def _active_extension_video_frames(self) -> int:
        return self._extension_video_frames if self._step_recipe in (None, "extension") else 0

    @property
    def _active_extension_audio_latents(self) -> int:
        return self._extension_audio_latents if self._step_recipe in (None, "extension") else 0

    supports_validation = True

    def __init__(self):
        super().__init__()
        self.backend: H3TrainingBackend | None = None
        self._crepa_config: H3CREPAConfig | None = None
        self._guidance_probability_generator: torch.Generator | None = None
        self._guidance_scale_generator: torch.Generator | None = None
        self._guidance_scale_range: tuple[float, float] | None = None
        self._recipe_probability_generator: torch.Generator | None = None
        self._qwen_control_dropout_generator: torch.Generator | None = None
        self._overlay_network = None
        self._h3_profiler: H3StepProfiler | None = None
        # The frozen base's field is a property of the checkpoint and the validation
        # item, not of the training run, so it is measured once and kept.
        self._field_base_gaps: dict[tuple, float] = {}
        self._field_base_branches: dict[tuple, tuple] = {}
        # The optimizer step the running validation belongs to, and whether the
        # base reference of a full fine-tune has already been read from its file.
        self._validation_global_step: int | None = None
        self._field_probe_snapshot_loaded = False
        self._field_ratios: dict[int, list[float]] = {}
        self._field_cosines: dict[int, list[float]] = {}
        self._field_distances: dict[int, list[float]] = {}
        self._null_field_ratios: dict[int, list[float]] = {}
        self._velocity_errors: dict[int, list[float]] = {}
        self._velocity_error_ratios: dict[int, list[float]] = {}
        self._branch_drift: dict[int, list] = {}
        self._prompted_drift_ratios: dict[int, list[float]] = {}
        self._validation_network = None
        self._adapter_network = None
        self._adapter_prompt_only = False
        self._measured_variance_curve = None
        self._validation_multipliers = []
        self._adapter_ema = None
        self._step_recipe: str | None = None
        self._step_qwen_control_dropout = False
        self._crepa: H3CREPA | None = None
        self._extension_video_frames = 0
        self._extension_audio_latents = 0
        self._extension_route = "condition_rows"
        self._frame_sigma_jitter = 0.0
        self._step_row_video_timestep = None
        self._spatial_density_jitter = 0.0
        self._step_spatial_density_scale = None
        self._keyframe_anchors: tuple[int | str, ...] = ()
        self._keyframe_random_count = 0
        self._mask_mode = "off"
        self._mask_audio = False
        self._mask_bounds = (0.25, 0.75)
        self._step_keyframes = None
        self._step_guides = None
        self._step_reference_modality = "av"
        self._step_mask = None
        self._validation_dataloader = None
        self._rollout_probability_generator: torch.Generator | None = None
        self._preservation_probability_generator: torch.Generator | None = None
        self._dop_probability_generator: torch.Generator | None = None
        self._rollout_noise_generator: torch.Generator | None = None
        self._rollout_teacher: H3RolloutTeacherCache | None = None
        self._rollout_data_err: dict[int, list[float]] = {}
        self._rollout_field: list[float] = []
        self._rollout_field_cos: list[float] = []
        self._rollout_probe_done: set = set()

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module):
        del network_module
        if not weights_sd:
            return weights_sd
        first_key = next(iter(weights_sd))
        if first_key.startswith("lora_"):
            return weights_sd
        if first_key.startswith(("diffusion_model.", "transformer.")):
            logger.info("Converting MiniMax H3 base LoRA weights from Diffusers format")
            return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
        return weights_sd

    def load_network_weights(self, path: str, network_module_name: str) -> dict[str, torch.Tensor]:
        return self.convert_weight_keys(load_file(path), network_module_name)

    def _build_dataset(self, args):
        if args.num_timestep_buckets is not None:
            logger.info("Using timestep bucketing. Number of buckets: %s", args.num_timestep_buckets)
        self.num_timestep_buckets = args.num_timestep_buckets
        current_epoch = Value("i", 0)

        logger.info("Load dataset config from %s", args.dataset_config)
        user_config = config_utils.load_user_config(args.dataset_config)
        _validate_dataset_loss_coverage(
            user_config,
            video_weight=float(getattr(args, "h3_video_loss_weight", 1.0)),
            audio_weight=float(getattr(args, "h3_audio_loss_weight", 1.0)),
        )
        train_dataset_group, _ = create_h3_dataset_group(
            user_config,
            args,
            training=True,
            num_timestep_buckets=self.num_timestep_buckets,
            shared_epoch=current_epoch,
        )
        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
            )

        if getattr(args, "h3_rollout_supervision", False):
            # Built here, before the transformer is loaded, because an unpaired or
            # control-free teacher corpus must fail in seconds rather than at the
            # first active step, which at --h3_rollout_probability 0.5 is minutes
            # into the run.
            enable_item_keys(train_dataset_group)
            teacher_config = config_utils.load_user_config(args.h3_rollout_teacher_config)
            teacher_group, _ = create_h3_dataset_group(
                teacher_config,
                args,
                training=True,
                # The teacher supplies conditioning, never a schedule: its own
                # bucketing would only build a timestep pool nothing reads.
                num_timestep_buckets=None,
                shared_epoch=Value("i", 0),
            )
            self._rollout_teacher = H3RolloutTeacherCache.from_dataset_group(teacher_group)
            self._rollout_teacher.require(item_key_text_caches(train_dataset_group))
            # The student's latent caches are read here only for their reference
            # counts: "privileged" on that channel means the teacher holds more
            # references than the student for the same item, which is a
            # comparison and not a property of either arm alone.
            channel, paired = self._rollout_teacher.validate_privilege(
                item_key_latent_caches(train_dataset_group),
                channel=getattr(args, "h3_rollout_teacher_privilege", "auto"),
                student_text_paths=item_key_text_caches(train_dataset_group),
            )
            # Named per channel rather than as a binary: this line is the only
            # place a run says which privilege it actually resolved, and a label
            # that quietly describes the wrong one turns the run's whole premise
            # into a guess for whoever reads the log later.
            privilege = {
                "qwen": "Qwen control visuals",
                "reference": "extra reference latents (Ref2VA variant B)",
                "keyframe": "an endpoint conditioning task the student does not declare",
                "caption": "a longer caption for the same clip than the student is shown",
            }[channel]
            logger.info(
                "H3 rollout supervision paired %d teacher items, all privileged through %s, from %s",
                paired,
                privilege,
                args.h3_rollout_teacher_config,
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)
        return train_dataset_group, collator, current_epoch

    def _build_validation_dataloader(self, args, accelerator):
        validation_seed = args.validation_seed if args.validation_seed is not None else args.seed
        with preserve_rng_state():
            seed_validation_forward(validation_seed)
            validation_args = copy.copy(args)
            validation_args.h3_load_dino_features = False
            current_epoch = Value("i", 0)
            user_config = config_utils.load_user_config(args.validation_dataset_config)
            dataset_group, _ = create_h3_dataset_group(
                user_config,
                validation_args,
                training=True,
                num_timestep_buckets=None,
                shared_epoch=current_epoch,
            )
        if dataset_group.num_train_items == 0 or len(dataset_group) == 0:
            raise ValueError("MiniMax H3 validation dataset contains no cached items")
        item_count = len(dataset_group)
        if args.max_validation_items is not None:
            item_count = min(item_count, args.max_validation_items)
        indices = range(accelerator.process_index, item_count, accelerator.num_processes)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(validation_seed)
        return torch.utils.data.DataLoader(
            _IndexedValidationDataset(dataset_group, indices),
            batch_size=None,
            num_workers=0,
            generator=loader_generator,
        )

    _VALIDATION_POOLS = (
        "_field_ratios",
        "_field_cosines",
        "_field_distances",
        "_null_field_ratios",
        "_velocity_errors",
        "_velocity_error_ratios",
        "_rollout_data_err",
        "_branch_drift",
        "_prompted_drift_ratios",
    )

    def _validation_pools(self) -> dict:
        return {name: getattr(self, name) for name in self._VALIDATION_POOLS} | {
            "_rollout_field": self._rollout_field,
            "_rollout_field_cos": self._rollout_field_cos,
            "_rollout_probe_done": self._rollout_probe_done,
        }

    def _restore_validation_pools(self, pools: dict) -> None:
        for name, value in pools.items():
            setattr(self, name, value)

    def _reset_validation_pools(self) -> None:
        for name in self._VALIDATION_POOLS:
            setattr(self, name, {})
        self._rollout_field = []
        self._rollout_field_cos = []
        self._rollout_probe_done = set()

    @staticmethod
    def _pooled(values: dict, accelerator=None) -> float | None:
        """Mean over every pooled value, across ranks when an accelerator is given."""
        pooled = [item for items in values.values() for item in items]
        if accelerator is None:
            return sum(pooled) / len(pooled) if pooled else None
        stats = torch.tensor([float(sum(pooled)), float(len(pooled))], dtype=torch.float64, device=accelerator.device)
        reduce = getattr(accelerator, "reduce", None)
        if callable(reduce):
            stats = reduce(stats, reduction="sum")
        return float(stats[0] / stats[1]) if float(stats[1]) > 0 else None

    def _validate_multiplier_sweep(
        self, accelerator, args, transformer, network, bins, observed_modes, validation_tasks, validation_seed
    ) -> dict[str, float]:
        """Repeat the data-state validation pass at other adapter multipliers.

        The adapter's strength is the inference knob every user turns; this is how
        the field and the fit respond to it, the analogue of a guidance-scale
        sensitivity curve. Only the per-batch data-state probes are pooled (loss
        accumulators and the rollout probe are not repeated); the live pools are
        put back untouched afterwards.
        """
        multipliers = [m for m in self._validation_multipliers if m != 1.0]
        if not multipliers or network is None:
            return {}
        unwrapped = accelerator.unwrap_model(network)
        modules = list(getattr(unwrapped, "unet_loras", ())) + list(getattr(unwrapped, "text_encoder_loras", ()))
        if not modules:
            logger.warning("--h3_validation_multipliers: the network exposes no LoRA modules to scale; sweep skipped")
            return {}
        original = [module.multiplier for module in modules]
        saved = self._validation_pools()
        probe_steps = getattr(args, "h3_validation_rollout_probe", 0)
        metrics: dict[str, float] = {}
        try:
            for multiplier in multipliers:
                for module in modules:
                    module.multiplier = multiplier
                self._reset_validation_pools()
                # The rollout probe is expensive and not repeated: mark it done.
                self._rollout_probe_done = None
                accumulators = {
                    task: H3ValidationAccumulator(
                        len(bins),
                        balance=args.h3_loss_balance,
                        video_weight=0.0 if task[0] == "video" else args.h3_video_loss_weight,
                        audio_weight=0.0 if task[0] == "audio" else args.h3_audio_loss_weight,
                    )
                    for task in validation_tasks
                }
                args.h3_validation_rollout_probe = 0
                with preserve_rng_state():
                    for dataset_index, batch in self._validation_dataloader:
                        self._validate_batch(
                            accelerator,
                            args,
                            transformer,
                            dataset_index,
                            batch,
                            bins,
                            observed_modes,
                            validation_tasks,
                            accumulators,
                            validation_seed,
                        )
                tag = f"val/m{multiplier:g}"
                for key, pool in (
                    ("velocity_err_rel", self._velocity_error_ratios),
                    ("field", self._field_ratios),
                    ("field_cos", self._field_cosines),
                    ("drift/prompted_rel", self._prompted_drift_ratios),
                ):
                    value = self._pooled(pool, accelerator)
                    if value is not None:
                        metrics[f"{tag}/{key}"] = value
        finally:
            args.h3_validation_rollout_probe = probe_steps
            for module, value in zip(modules, original, strict=True):
                module.multiplier = value
            self._restore_validation_pools(saved)
        return metrics

    @torch.no_grad()
    def validate(
        self,
        accelerator,
        args,
        transformer,
        network,
        global_step,
        epoch,
    ) -> None:
        del epoch
        if self.backend is None:
            raise RuntimeError("H3 training backend is not loaded")
        # Discard conditioning from the last training step. Validation redraws
        # its configured mask/keyframes deterministically below; jitter and
        # spatial-density augmentation remain disabled for the canonical metric.
        self._step_mask = None
        self._step_row_video_timestep = None
        self._step_spatial_density_scale = None
        self._step_keyframes = None
        self._step_guides = None
        self._step_reference_modality = "av"
        self._step_qwen_control_dropout = False
        self._step_recipe = None
        if self._validation_dataloader is None:
            self._validation_dataloader = self._build_validation_dataloader(args, accelerator)

        bins = validation_sigma_bins(
            args.validation_timestep_bins,
            minimum=args.validation_min_timestep / 1000.0,
            maximum=args.validation_max_timestep / 1000.0,
            video_shift=args.h3_shift_video,
            audio_shift=args.h3_shift_audio,
        )
        # Random observed-modality training optimizes three distinct tasks. A
        # single random validation draw would make successive measurements
        # incomparable, while reporting only the joint task would hide both
        # conditional directions. Evaluate every direction deterministically.
        if args.h3_observed_modality == "random":
            observed_modes = [None]
            if args.h3_audio_loss_weight > 0:
                observed_modes.append("video")
            if args.h3_video_loss_weight > 0:
                observed_modes.append("audio")
            observed_modes = tuple(observed_modes)
        else:
            observed_modes = (args.h3_observed_modality,)
        validation_tasks = tuple((observed, reference) for observed in observed_modes for reference in ("av", "video", "audio"))
        accumulators = {
            task: H3ValidationAccumulator(
                len(bins),
                balance=args.h3_loss_balance,
                video_weight=0.0 if task[0] == "video" else args.h3_video_loss_weight,
                audio_weight=0.0 if task[0] == "audio" else args.h3_audio_loss_weight,
            )
            for task in validation_tasks
        }
        validation_seed = args.validation_seed if args.validation_seed is not None else args.seed
        sweep_metrics: dict[str, float] = {}
        self._validation_network = network
        self._adapter_prompt_only = bool(getattr(args, "h3_adapter_prompt_only", False))
        self._validation_global_step = int(global_step)
        if (
            getattr(args, "h3_validation_field_probe", False)
            and int(global_step) > 0
            and self._probe_base_is_the_live_model(accelerator, network)
        ):
            # A resumed full fine-tune: its weights are no longer the checkpoint, so
            # the step-zero reference has to come from the file. At step 0 nothing is
            # read -- the reference is taken live and the file rewritten -- so a run
            # that reuses an output name cannot inherit a previous run's reference.
            if accelerator.num_processes != 1:
                raise ValueError(
                    "--h3_validation_field_probe on a full fine-tune keeps its step-zero reference in one process's "
                    "file; validation items are sharded per process, so it supports a single process only"
                )
            self._load_field_probe_snapshot(self._field_probe_snapshot_path(args), self._field_probe_fingerprint(args))
        self._field_ratios = {}
        self._field_cosines = {}
        self._field_distances = {}
        self._null_field_ratios = {}
        self._velocity_errors = {}
        self._velocity_error_ratios = {}
        self._rollout_data_err = {}
        self._rollout_field = []
        self._rollout_field_cos = []
        self._rollout_probe_done = set()
        self._branch_drift = {}
        self._prompted_drift_ratios = {}

        block_swap_active = bool(self.blocks_to_swap)
        transformer_was_training = transformer.training
        network_was_training = network.training if network is not None else None
        try:
            transformer.eval()
            if network is not None:
                # LoRA modules are registered below the network but invoked by
                # transformer forwards. Without this, network/rank/module
                # dropout remains active and validation measures a regularized
                # training draw rather than the saved adapter.
                network.eval()
            if block_swap_active:
                # Validation has no backward pass. A training-mode offloader
                # leaves the swapped prefix on CPU because it expects backward
                # hooks to restore it before the next forward.
                transformer.switch_block_swap_for_inference()
            use_ema = bool(getattr(args, "h3_validate_ema", False))
            with self._adapter_ema_weights(accelerator, network) if use_ema else nullcontext() as ema_swapped:
                if use_ema and not ema_swapped:
                    logger.warning("--h3_validate_ema: no EMA yet (first validation before any optimizer step); validating live")
                with preserve_rng_state():
                    for dataset_index, batch in self._validation_dataloader:
                        self._validate_batch(
                            accelerator,
                            args,
                            transformer,
                            dataset_index,
                            batch,
                            bins,
                            observed_modes,
                            validation_tasks,
                            accumulators,
                            validation_seed,
                        )
                sweep_metrics = self._validate_multiplier_sweep(
                    accelerator, args, transformer, network, bins, observed_modes, validation_tasks, validation_seed
                )
        finally:
            self._step_mask = None
            self._step_keyframes = None
            self._step_guides = None
            self._step_reference_modality = "av"
            self._step_qwen_control_dropout = False
            self._step_recipe = None
            if block_swap_active:
                transformer.switch_block_swap_for_training()
            transformer.train(transformer_was_training)
            if network is not None:
                network.train(network_was_training)

        metrics = {}
        metrics.update(sweep_metrics)
        observed_labels = {None: "joint", "video": "v2a", "audio": "a2v"}
        reduced_metrics = {}
        for (observed, reference), accumulator in accumulators.items():
            reduced = accelerator.reduce(accumulator.reduction_tensor(device=accelerator.device), reduction="sum")
            accumulator.load_reduced_tensor(reduced)
            task_metrics = accumulator.metrics()
            if task_metrics:
                reduced_metrics[(observed, reference)] = task_metrics
        active_references = {reference for _, reference in reduced_metrics}
        active_observed = {observed for observed, _ in reduced_metrics}
        for (observed, reference), task_metrics in reduced_metrics.items():
            if len(active_observed) == 1 and active_references == {"av"}:
                prefix = "val"
            else:
                prefix = f"val/{observed_labels[observed]}"
                if active_references != {"av"}:
                    prefix += f"/ref_{reference}"
            metrics.update({f"{prefix}/{key}": value for key, value in task_metrics.items()})
        # Reported per bin as well as pooled: the field is not lost uniformly, and the
        # high-noise end -- where composition and prompt following are decided -- is
        # both the first to go and the one worth watching.
        for bin_index, ratios in sorted(self._field_ratios.items()):
            if ratios:
                metrics[f"val/field/bin{bin_index}"] = sum(ratios) / len(ratios)
        pooled = [ratio for ratios in self._field_ratios.values() for ratio in ratios]
        if pooled:
            metrics["val/field"] = sum(pooled) / len(pooled)
        # The ratio is a length, and guidance is a vector. An adapter can hold the
        # length exactly while turning the direction the field pushes in, which the
        # ratio alone would report as a field left untouched, so the two are only
        # meaningful read together: a field survives when both are near 1.
        for bin_index, cosines in sorted(self._field_cosines.items()):
            if cosines:
                metrics[f"val/field_cos/bin{bin_index}"] = sum(cosines) / len(cosines)
        for bin_index, ratios in sorted(self._velocity_error_ratios.items()):
            if ratios:
                metrics[f"val/velocity_err_rel/bin{bin_index}"] = sum(ratios) / len(ratios)
        pooled_rel = [ratio for ratios in self._velocity_error_ratios.values() for ratio in ratios]
        if pooled_rel:
            metrics["val/velocity_err_rel"] = sum(pooled_rel) / len(pooled_rel)
        pooled_cos = [cosine for cosines in self._field_cosines.values() for cosine in cosines]
        if pooled_cos:
            metrics["val/field_cos"] = sum(pooled_cos) / len(pooled_cos)
        # Reported beside the two it combines, not instead of them: the pair says HOW a
        # field was lost -- shortened, turned, or both -- while this says how far it went.
        for bin_index, distances in sorted(self._field_distances.items()):
            if distances:
                metrics[f"val/field_dist/bin{bin_index}"] = sum(distances) / len(distances)
        pooled_dist = [distance for distances in self._field_distances.values() for distance in distances]
        if pooled_dist:
            metrics["val/field_dist"] = sum(pooled_dist) / len(pooled_dist)
        # Per step as well as pooled: the question is not only how wrong the walk ends
        # up but whether the error compounds, which is what a single-step metric cannot
        # see by construction.
        for step_index, errors in sorted(self._rollout_data_err.items()):  # x0 estimate error along the walk
            if errors:
                metrics[f"val/rollout/x0_err/step{step_index}"] = sum(errors) / len(errors)
        pooled_rollout_data = [error for errors in self._rollout_data_err.values() for error in errors]
        if pooled_rollout_data:
            metrics["val/rollout/x0_err"] = sum(pooled_rollout_data) / len(pooled_rollout_data)
        if self._rollout_field:
            metrics["val/rollout/field"] = sum(self._rollout_field) / len(self._rollout_field)
        if self._rollout_field_cos:
            metrics["val/rollout/field_cos"] = sum(self._rollout_field_cos) / len(self._rollout_field_cos)
        # Error against the raw data velocity, which every arm can be judged by no
        # matter what it optimised. val/loss cannot do that job: a guidance loss, a
        # teacher-matching loss and a rollout objective each report on their own
        # scale, so two arms configured differently are only comparable through a
        # separate offline evaluation. This comes from a forward the probe already
        # ran, which lets any run be placed against any other from the log alone.
        for bin_index, errors in sorted(self._velocity_errors.items()):
            if errors:
                metrics[f"val/velocity_err/bin{bin_index}"] = sum(errors) / len(errors)
        pooled_err = [error for errors in self._velocity_errors.values() for error in errors]
        if pooled_err:
            metrics["val/velocity_err"] = sum(pooled_err) / len(pooled_err)
        # What the field would be with the empty branch held where the base left it
        # and the prompted branch wherever training took it.
        #
        # Read it as an amplitude and nothing more. It is a distance from a fixed point,
        # so it cannot separate "the prompted branch stayed put" from "it went somewhere
        # else and happened to land equally far away" -- and measured on real arms it
        # does invert: ordinary training scores HIGHER here than an arm whose prompted
        # branch demonstrably moved less. For the question "does this adapter still
        # answer prompts the way the checkpoint did", read val/drift/prompted_rel.
        counterfactual = [ratio for ratios in self._null_field_ratios.values() for ratio in ratios]
        if counterfactual:
            metrics["val/field_if_null_pinned"] = sum(counterfactual) / len(counterfactual)
        drifts = [pair for pairs in self._branch_drift.values() for pair in pairs]
        if drifts:
            metrics["val/drift/prompted"] = sum(pair[0] for pair in drifts) / len(drifts)
            metrics["val/drift/empty"] = sum(pair[1] for pair in drifts) / len(drifts)
        for bin_index, ratios in sorted(self._prompted_drift_ratios.items()):
            if ratios:
                metrics[f"val/drift/prompted_rel/bin{bin_index}"] = sum(ratios) / len(ratios)
        pooled_drift = [ratio for ratios in self._prompted_drift_ratios.values() for ratio in ratios]
        if pooled_drift:
            metrics["val/drift/prompted_rel"] = sum(pooled_drift) / len(pooled_drift)
        if metrics and len(accelerator.trackers) > 0:
            accelerator.log(metrics, step=global_step)
        accelerator.print("MiniMax H3 validation: " + ", ".join(f"{key}={value:.6g}" for key, value in metrics.items()))
        if (
            getattr(args, "h3_validation_field_probe", False)
            and int(global_step) == 0
            and self._field_base_branches
            and self._probe_base_is_the_live_model(accelerator, network)
            and accelerator.num_processes == 1
        ):
            self._save_field_probe_snapshot(self._field_probe_snapshot_path(args), self._field_probe_fingerprint(args))

    @staticmethod
    def _probe_base_is_the_live_model(accelerator, network) -> bool:
        """True when nothing can be switched off to reach the frozen base.

        A LoRA run reaches the checkpoint by disabling its network for one forward.
        A full fine-tune trains the checkpoint itself: its "network" is the dense
        module, which has no ``set_enabled``, and the only moment its weights ARE
        the base is step zero.
        """
        if network is None:
            return True
        unwrap = getattr(accelerator, "unwrap_model", None)
        module = unwrap(network) if callable(unwrap) else network
        return not callable(getattr(module, "set_enabled", None))

    @staticmethod
    def _field_probe_snapshot_path(args) -> str:
        return os.path.join(args.output_dir, f"{args.output_name}_field_probe_base.pt")

    @staticmethod
    def _field_probe_fingerprint(args) -> dict[str, str]:
        """What the stored reference is a reference FOR.

        The cache keys name an item by its position in the validation set and a sigma
        by its bin, and both are meaningless under another validation config, seed,
        item cap or checkpoint. The fingerprint records those so a file left by a
        different configuration is refused rather than read as if it matched.
        """

        def file_digest(path):
            if not path or not os.path.exists(path):
                return str(path)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def file_identity(path):
            if not path or not os.path.exists(path):
                return str(path)
            stat = os.stat(path)
            return f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"

        return {
            "dit": file_identity(getattr(args, "dit", None)),
            "validation_dataset_config": file_digest(getattr(args, "validation_dataset_config", None)),
            "validation_seed": str(args.validation_seed if getattr(args, "validation_seed", None) is not None else args.seed),
            "max_validation_items": str(getattr(args, "max_validation_items", None)),
            "h3_training_mode": str(getattr(args, "h3_training_mode", None)),
        }

    def _save_field_probe_snapshot(self, path: str, fingerprint: dict[str, str]) -> None:
        """Keep the step-zero base reference beside the checkpoints, for resumes."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"fingerprint": dict(fingerprint), "gaps": dict(self._field_base_gaps), "branches": dict(self._field_base_branches)},
            path,
        )
        logger.info("MiniMax H3 field probe: saved the frozen-base reference of %d items to %s", len(self._field_base_gaps), path)

    def _load_field_probe_snapshot(self, path: str, fingerprint: dict[str, str]) -> None:
        if self._field_probe_snapshot_loaded or self._field_base_gaps:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--h3_validation_field_probe on a resumed full fine-tune needs the step-zero reference {path}, "
                "which this run did not write; start it over with --validate_at_start, or drop the probe"
            )
        # Tensors, strings, numbers and tuple keys only, so the safe unpickler suffices.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        stored = payload.get("fingerprint", {})
        mismatched = sorted(name for name in set(stored) | set(fingerprint) if stored.get(name) != fingerprint.get(name))
        if mismatched:
            raise ValueError(
                f"the field-probe reference {path} was taken under a different {', '.join(mismatched)}; "
                "its per-item pairs do not describe this validation, so it is refused"
            )
        self._field_base_gaps.update(payload["gaps"])
        self._field_base_branches.update(payload["branches"])
        self._field_probe_snapshot_loaded = True
        logger.info("MiniMax H3 field probe: read the frozen-base reference of %d items from %s", len(self._field_base_gaps), path)

    def _validate_batch(
        self,
        accelerator,
        args,
        transformer,
        dataset_index,
        batch,
        bins,
        observed_modes,
        validation_tasks,
        accumulators,
        validation_seed,
    ) -> None:
        latents = self.get_primary_latents(batch)
        if latents.shape[0] != 1:
            raise ValueError("MiniMax H3 validation requires dataset batch_size = 1")
        has_video = "latents" in batch or latents.ndim == 5
        has_audio = H3_AUDIO_LATENTS_KEY in batch
        video_source = batch.get("latents", latents if latents.ndim == 5 else None)
        video_latents = video_source.to(accelerator.device, dtype=self.dit_dtype) if has_video else None
        audio_latents = batch[H3_AUDIO_LATENTS_KEY].to(accelerator.device, dtype=self.dit_dtype) if has_audio else None
        spatial_tokens = bool(args.h3_audio_only_spatial_tokens and not has_video and has_audio)
        if spatial_tokens:
            video_latents = self._build_audio_only_spatial_tokens(audio_latents)
        is_image = has_video and not has_audio and video_latents.shape[2] == 1
        if len(observed_modes) == 1 and observed_modes[0] is not None and not (has_video and has_audio):
            raise ValueError("H3 observed-modality validation requires cached video and audio targets")

        batch_observed_modes = list(observed_modes)
        if len(observed_modes) > 1:
            valid_video = self._target_has_valid_elements(batch, "video_loss_mask", has_video)
            valid_audio = self._target_has_valid_elements(batch, "audio_loss_mask", has_audio)
            batch_observed_modes = []
            if (valid_video and args.h3_video_loss_weight > 0) or (valid_audio and args.h3_audio_loss_weight > 0):
                batch_observed_modes.append(None)
            if has_video and has_audio and valid_audio and args.h3_audio_loss_weight > 0:
                batch_observed_modes.append("video")
            if has_video and has_audio and valid_video and args.h3_video_loss_weight > 0:
                batch_observed_modes.append("audio")

        probabilities = batch.get(H3_REFERENCE_MODALITY_PROBABILITIES_KEY)
        batch_reference_modes = ("av",)
        if probabilities is not None:
            if isinstance(probabilities, (list, tuple)):
                if len(probabilities) != 1:
                    raise ValueError("H3 validation reference modality probabilities must contain one batch item")
                probabilities = probabilities[0]
            if probabilities.ndim == 2 and probabilities.shape[0] == 1:
                probabilities = probabilities[0]
            probabilities = probabilities.detach().to(device="cpu", dtype=torch.float32)
            if probabilities.shape != (3,):
                raise ValueError("H3 validation reference modality probabilities must have shape [3]")
            batch_reference_modes = tuple(
                modality for modality, probability in zip(("av", "video", "audio"), probabilities) if float(probability) > 0
            )

        for sigma_bin in bins:
            if video_latents is not None:
                video_noise_seed = derive_validation_seed(
                    validation_seed,
                    dataset_index=dataset_index,
                    bin_index=sigma_bin.index,
                    stream="video-noise",
                )
                seed_validation_forward(video_noise_seed)
                video_noise = torch.randn_like(video_latents)
            else:
                video_noise = None
            if audio_latents is not None:
                audio_noise_seed = derive_validation_seed(
                    validation_seed,
                    dataset_index=dataset_index,
                    bin_index=sigma_bin.index,
                    stream="audio-noise",
                )
                seed_validation_forward(audio_noise_seed)
                audio_noise = torch.randn_like(audio_latents)
            else:
                audio_noise = None

            base_sigma = torch.tensor([sigma_bin.base_sigma], device=accelerator.device, dtype=torch.float32)
            if is_image:
                base_sigma = image_validation_sigma(
                    base_sigma,
                    latent_height=video_latents.shape[-2],
                    latent_width=video_latents.shape[-1],
                    flow_shift=args.h3_image_flow_shift,
                )
            for observed in batch_observed_modes:
                for reference in batch_reference_modes:
                    task = (observed, reference)
                    if task not in validation_tasks:
                        continue
                    self._step_reference_modality = reference
                    self._validate_observed_variant(
                        accelerator,
                        args,
                        transformer,
                        dataset_index,
                        batch,
                        sigma_bin,
                        observed,
                        video_latents,
                        audio_latents,
                        video_noise,
                        audio_noise,
                        base_sigma,
                        is_image,
                        accumulators[task],
                        validation_seed,
                        spatial_tokens,
                    )
            self._step_reference_modality = "av"

    def _validate_observed_variant(
        self,
        accelerator,
        args,
        transformer,
        dataset_index,
        batch,
        sigma_bin,
        observed,
        video_latents,
        audio_latents,
        video_noise,
        audio_noise,
        base_sigma,
        is_image,
        accumulator,
        validation_seed,
        spatial_tokens,
    ) -> None:
        inputs = prepare_joint_noisy_inputs(
            video_latents,
            audio_latents,
            video_noise,
            audio_noise,
            base_sigma,
            video_shift=1.0 if is_image else args.h3_shift_video,
            audio_shift=1.0 if is_image else args.h3_shift_audio,
            observed=observed,
        )
        video_weight = 0.0 if observed == "video" or spatial_tokens else args.h3_video_loss_weight
        audio_weight = 0.0 if observed == "audio" else args.h3_audio_loss_weight

        # Validation uses the configured task, not whichever random mask or
        # keyframes survived from the last train step. Re-draw them
        # deterministically and use the same effective loss mask as training.
        conditioning_seed = derive_validation_seed(
            validation_seed,
            dataset_index=dataset_index,
            # Keep conditioning fixed across sigma bins and observed variants,
            # so their metrics isolate the intended axis of comparison.
            bin_index=0,
            stream="conditioning",
        )
        seed_validation_forward(conditioning_seed)
        self._step_keyframes = self._resolve_keyframe_anchors(inputs.video)
        self._step_guides = self._resolve_guide_specs(inputs.video, inputs.audio)
        # Validation measures one fixed recipe so successive numbers stay
        # comparable; a run that mixes masking and extension per step reports the
        # masked one, since the two cannot share a step.
        self._step_recipe = "mask" if all(self._configured_recipes()) else None
        self._step_mask = self._draw_step_mask(inputs, tuple(VIDEO_DIT_PATCH_SIZE), batch)
        effective_video_mask = self._mask_to_loss(
            self._extension_masked(batch.get("video_loss_mask"), inputs.video_target, self._active_extension_video_frames, axis=-3),
            inputs.video_target,
            None if self._step_mask is None else self._step_mask.video_latent,
            axis=-3,
        )
        effective_audio_mask = self._mask_to_loss(
            self._extension_masked(
                batch.get("audio_loss_mask"), inputs.audio_target, self._active_extension_audio_latents, axis=-1
            ),
            inputs.audio_target,
            None if self._step_mask is None else self._step_mask.audio_latent,
            axis=-1,
        )

        forward_seed = derive_validation_seed(
            validation_seed,
            dataset_index=dataset_index,
            bin_index=sigma_bin.index,
            stream="model-forward",
        )
        seed_validation_forward(forward_seed)
        if args.h3_guidance_distillation_scale is not None:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError("guidance-consistent H3 validation is missing " + ", ".join(missing_empty))
            fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
            # Mirror the training empty branch: it is an auxiliary forward, so it
            # must use the same INT8-attention calibration or validation measures
            # a different model than training optimizes.
            int8_context = getattr(transformer, "int8_attention_context", None)
            with (
                torch.random.fork_rng(devices=fork_devices),
                int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
            ):
                empty_prediction = self._predict(
                    accelerator,
                    transformer,
                    batch,
                    inputs,
                    conditioning="empty",
                )
        prediction = self._predict(
            accelerator,
            transformer,
            batch,
            inputs,
            conditioning="prompt",
        )
        if args.h3_guidance_distillation_scale is not None:
            prediction, loss_inputs = self._guidance_loss_inputs(args, prediction, empty_prediction, inputs)
        else:
            loss_inputs = inputs

        video_sample_weight = self._sample_weight(args, inputs.video_sigma) if video_latents is not None else None
        audio_sample_weight = self._sample_weight(args, inputs.audio_sigma, modality="audio") if audio_latents is not None else None
        if prediction.video is not None and loss_inputs.video_target is not None and video_weight > 0:
            total, count = masked_squared_error_sum(
                prediction.video,
                loss_inputs.video_target,
                effective_video_mask,
                sample_weight=video_sample_weight,
            )
            accumulator.add(sigma_bin.index, "video", total, count)
        if prediction.audio is not None and loss_inputs.audio_target is not None and audio_weight > 0:
            total, count = masked_squared_error_sum(
                prediction.audio,
                loss_inputs.audio_target,
                effective_audio_mask,
                sample_weight=audio_sample_weight,
            )
            accumulator.add(sigma_bin.index, "audio", total, count)

        if int(getattr(args, "h3_validation_rollout_probe", 0) or 0) > 0:
            self._probe_rollout_field(
                accelerator,
                args,
                transformer,
                batch,
                effective_video_mask,
                dataset_index=dataset_index,
                observed=observed,
            )
        if getattr(args, "h3_validation_field_probe", False):
            self._probe_guidance_field(
                accelerator,
                args,
                transformer,
                batch,
                inputs,
                effective_video_mask,
                dataset_index=dataset_index,
                sigma_bin=sigma_bin,
                observed=observed,
            )

        self._step_mask = None
        self._step_keyframes = None
        self._step_guides = None
        self._step_recipe = None

    @staticmethod
    def _masked_cosine(lhs, rhs, mask):
        """Cosine between two fields over the authored elements only."""
        left = lhs.float().flatten()
        right = rhs.float().flatten()
        if mask is not None:
            valid = mask.to(device=lhs.device, dtype=torch.float32).expand_as(lhs).flatten()
            left = left * valid
            right = right * valid
        denominator = float(left.norm()) * float(right.norm())
        if denominator == 0.0:
            return 0.0
        return float(torch.dot(left, right)) / denominator

    @staticmethod
    def _masked_rms(tensor, mask):
        """Root mean square over the authored elements only."""
        if mask is None:
            return float(tensor.float().pow(2).mean().sqrt())
        valid = mask.to(device=tensor.device, dtype=torch.float32).expand_as(tensor)
        count = float(valid.sum())
        if count == 0.0:
            return 0.0
        return float((tensor.float().pow(2) * valid).sum().div(count).sqrt())

    @torch.no_grad()
    def _probe_rollout_field(
        self,
        accelerator,
        args,
        transformer,
        batch,
        video_mask,
        *,
        dataset_index,
        observed,
    ) -> None:
        """The field where the model actually ends up, not where the data is.

        Every field number this trainer reports so far is taken at a noised DATA
        state: a clip from the set, corrupted to some sigma. Generation never visits
        those states. It starts at noise and walks a trajectory of its own, and each
        step lands on whatever the previous step produced, errors included. A method
        whose whole mechanism is correcting that walk -- supervising the student where
        its own sampler goes rather than where the data is -- cannot show up in a
        single-step measurement on data states, however carefully that measurement is
        made. It is not that the metric is imprecise there; it is looking elsewhere.

        This probe rolls the adapted model out from pure noise under the prompt, and
        rolls the frozen base out from the identical noise, then reports two things at
        the states reached:

        ``val/rollout/x0_err`` -- how wrong the clean-clip estimate is at every state
        the adapted model's own walk visits, per step and pooled. Error that grows
        with the step index is compounding error, which is a different failure from a
        uniformly different model, and only the per-step shape tells them apart.

        ``val/rollout/field`` -- the prompted-to-empty gap of both models measured at
        the SAME state, the one the adapted model reached. Normalising there rather
        than at each model's own endpoint isolates the difference between the models
        from the difference between the states.

        Cost is ``2 * steps + 4`` no-grad forwards per validation, paid once per
        dataset and observed modality rather than per item and bin.
        """
        steps = int(getattr(args, "h3_validation_rollout_probe", 0) or 0)
        if steps <= 0:
            return
        video_latents = batch.get("latents")
        if video_latents is None:
            return
        key = (dataset_index, observed, self._step_reference_modality)
        if key in self._rollout_probe_done:
            return
        self._rollout_probe_done.add(key)
        network = self._validation_network
        if network is None:
            raise ValueError(
                "--h3_validation_rollout_probe requires a trainable network: it evaluates the frozen base at the state "
                "the adapted model walked to, which a full fine-tune has no frozen base to evaluate"
            )
        merged = getattr(self, "_merged_base_weight_paths", None)
        if merged:
            # Same reason as the data-state probe: the base rollout is produced by
            # switching the trainable network off, and an adapter merged at load
            # time cannot be switched off, so every ratio would be taken against
            # the checkpoint plus that adapter while being labelled the base.
            raise ValueError(
                "--h3_validation_rollout_probe rolls the frozen checkpoint out as its reference, but --base_weights "
                "merged " + ", ".join(merged) + " into it at load time, where nothing can switch them off; "
                "drop the probe or the base weights"
            )
        set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_validation_rollout_probe")
        int8_context = getattr(transformer, "int8_attention_context", None)
        fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []

        stop = float(getattr(args, "h3_validation_rollout_stop", 0.5))
        # Window 1 rather than 0: the schedule builder rejects an empty window, and
        # the one extra sigma past the stop is simply never stepped to. Indices 0..steps
        # carry the walk; index steps is where the field is read.
        base_sigmas = torch.tensor(rollout_base_sigmas(stop, steps, 1), dtype=torch.float32)
        video_sigmas = shift_sigma(base_sigmas, VIDEO_FLOW_SHIFT)
        audio_sigmas = shift_sigma(base_sigmas, AUDIO_FLOW_SHIFT)
        video_latents = video_latents.to(device=accelerator.device)

        def predict(state, index, conditioning):
            inputs = self._rollout_state(
                video=state,
                audio=None,
                video_sigma=video_sigmas[index : index + 1],
                audio_sigma=audio_sigmas[index : index + 1],
            )
            with (
                torch.random.fork_rng(devices=fork_devices),
                int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
            ):
                return self._predict(accelerator, transformer, batch, inputs, conditioning=conditioning).video

        def walk(enabled, start):
            set_enabled(enabled)
            try:
                state = start.clone()
                for index in range(steps):
                    velocity = predict(state, index, "prompt")
                    if velocity is None:
                        return state
                    state = euler_advance(state, velocity, float(video_sigmas[index]), float(video_sigmas[index + 1]))
                return state
            finally:
                set_enabled(True)

        # One noise draw shared by both walks: the trajectories must differ because
        # the models differ, not because they started apart.
        start = self._rollout_noise(video_latents)
        adapted_end = walk(True, start)
        walk(False, start)

        # What the model believes the clean clip is, asked at every state its own walk
        # reaches.
        #
        # This exists because the two axes this study measures disagree with what a
        # viewer sees. An arm whose single-step velocity error is indistinguishable from
        # the untrained checkpoint -- 0.52 against 0.51 -- beat that checkpoint 9.5 to
        # 6.0 in a blind comparison on held-out scenes. Both cannot be true of a metric
        # that captures what matters, and the difference between them is that generation
        # walks thirty steps while velocity_err reads one, at a state the walk never
        # visits.
        #
        # H3 predicts the data-pointing velocity v = x0 - eps at x_t = (1 - s) x0 + s eps,
        # so the clean estimate is recoverable exactly: x0_hat = x_t + s * v. That makes
        # a ground truth available at every point of a rollout, without the rollout
        # needing to end anywhere in particular -- and x0_hat is what the sampler is
        # really steering, so being wrong about it is what a viewer eventually sees.
        #
        # Measured on the ADAPTED model along ITS OWN trajectory. Both halves matter: a
        # walk down the base's states would ask a question about the base, and reading
        # x0_hat at a noised data state would be velocity_err again in different units.
        latents = video_latents.float()
        set_enabled(True)
        state = start.clone()
        for index in range(steps):
            velocity = predict(state, index, "prompt")
            if velocity is None:
                break
            sigma_now = float(video_sigmas[index])
            estimate = state + sigma_now * velocity.float()
            energy = self._masked_rms(latents, video_mask)
            if energy > 0.0:
                self._rollout_data_err.setdefault(index, []).append(self._masked_rms(estimate - latents, video_mask) / energy)
            state = euler_advance(state, velocity, sigma_now, float(video_sigmas[index + 1]))

        def field_at(state, enabled):
            set_enabled(enabled)
            try:
                prompted = predict(state, steps, "prompt")
                empty = predict(state, steps, "empty")
            finally:
                set_enabled(True)
            return None if prompted is None or empty is None else prompted - empty

        adapted_field = field_at(adapted_end, True)
        base_field = field_at(adapted_end, False)
        if adapted_field is None or base_field is None:
            return
        base_size = self._masked_rms(base_field, video_mask)
        if base_size <= 0.0:
            return
        self._rollout_field.append(self._masked_rms(adapted_field, video_mask) / base_size)
        self._rollout_field_cos.append(self._masked_cosine(adapted_field, base_field, video_mask))

    @torch.no_grad()
    def _probe_guidance_field(
        self,
        accelerator,
        args,
        transformer,
        batch,
        inputs,
        video_mask,
        *,
        dataset_index,
        sigma_bin,
        observed,
    ) -> None:
        """How much of the base model's prompted-to-empty field the adapter still carries.

        A LoRA trained on a guidance-distilled checkpoint is not constrained to keep
        the difference between its prompted and its empty-prompt prediction, and that
        difference is what the checkpoint answers prompts with. Losing it is invisible
        in the training loss -- the loss goes down either way -- and shows up only
        later, as prompts being ignored and as detail the model had to invent falling
        apart. Reporting it during training turns a post-hoc audit into a curve.

        The quantity is the ratio of the adapter's field to the frozen base's, on the
        same items at the same noise: 1.0 means untouched, 0 means erased. The
        denominator does not change while training runs, so it is measured once and
        cached; after the first validation the probe costs two no-grad forwards per
        item and bin.

        Both branches are recomputed here rather than reused from the caller. The
        prompted prediction there has already been rewritten by the guidance loss when
        that objective is active, and a probe that silently measures a different
        quantity depending on which loss is configured would be worse than no probe.
        """
        if inputs.video is None:
            return
        missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
        if missing_empty:
            raise KeyError(
                "--h3_validation_field_probe compares the prompted branch against the empty one and therefore "
                "requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
            )
        network = self._validation_network
        # A full fine-tune has no network to switch off: the checkpoint it started from
        # exists only at step zero, so the base pair is taken then and kept (and written
        # beside the checkpoints, see _save_field_probe_snapshot). Later validations
        # reuse it, and an item without a stored pair cannot be measured at all.
        merged = getattr(self, "_merged_base_weight_paths", None)
        if merged:
            # Every number this probe reports is a ratio against the checkpoint, formed
            # by disabling the trainable network for one pair of forwards. --base_weights
            # is folded into the transformer at load time and cannot be switched off, so
            # the reference silently becomes "the checkpoint plus that adapter" and the
            # ratios land on a scale no other run shares -- while looking entirely
            # ordinary. Refusing is the only honest option: there is nothing to warn
            # about that a reader of the numbers could act on later.
            raise ValueError(
                "--h3_validation_field_probe measures against the checkpoint, but --base_weights merged "
                + ", ".join(merged)
                + " into it and a merge cannot be undone for one forward. Drop --base_weights, or drop the probe"
            )
        live_base = self._probe_base_is_the_live_model(accelerator, network)
        key = (dataset_index, sigma_bin.index, observed, self._step_reference_modality)
        if live_base and key not in self._field_base_gaps and int(getattr(self, "_validation_global_step", 0) or 0) != 0:
            raise ValueError(
                "--h3_validation_field_probe on a full fine-tune measures against a reference taken at step 0, and "
                f"none covers validation item {dataset_index} at sigma bin {sigma_bin.index}; start the run with "
                "--validate_at_start from the untouched checkpoint, or resume beside its <output_name>_field_probe_base.pt"
            )
        if live_base:

            def set_enabled(enabled: bool) -> None:
                del enabled

        else:
            set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_validation_field_probe")
        fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        int8_context = getattr(transformer, "int8_attention_context", None)

        def branches():
            # Both branches of one measurement must see the same stochastic
            # conditioning, or their difference reports the draw rather than the
            # prompt. Forking around each keeps the pair aligned and leaves the rest
            # of the validation's RNG stream untouched.
            #
            # The tensors are returned rather than reduced to a length here: the
            # angle between two fields, the counterfactual and the per-branch drift
            # all read them, and none of those can be recovered from a scalar.
            out = {}
            for branch in ("prompt", "empty"):
                with (
                    torch.random.fork_rng(devices=fork_devices),
                    int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
                ):
                    out[branch] = self._predict(accelerator, transformer, batch, inputs, conditioning=branch).video
            return out["prompt"], out["empty"]

        fresh_base = None
        if key not in self._field_base_gaps:
            set_enabled(False)
            try:
                base_prompted, base_empty = branches()
                self._field_base_gaps[key] = (
                    0.0 if base_prompted is None or base_empty is None else self._masked_rms(base_prompted - base_empty, video_mask)
                )
                # The base is fixed for the run, so keeping its two branches costs
                # the same one-off pair of forwards the denominator already paid.
                if base_prompted is not None and base_empty is not None:
                    self._field_base_branches[key] = (base_prompted.detach().cpu(), base_empty.detach().cpu())
                fresh_base = (base_prompted, base_empty)
            finally:
                set_enabled(True)

        if live_base and fresh_base is not None:
            # Step zero of a full fine-tune: the live model IS the base, so the pair
            # just taken is also the adapted pair, and two forwards are saved.
            adapted_prompted, adapted_empty = fresh_base
        else:
            adapted_prompted, adapted_empty = branches()
        adapted_pair = None
        if adapted_prompted is not None and adapted_empty is not None:
            adapted_pair = (adapted_prompted.detach().cpu(), adapted_empty.detach().cpu())
        base_pair = self._field_base_branches.get(key)
        if adapted_prompted is not None and inputs.video_target is not None:
            target = inputs.video_target.to(device=adapted_prompted.device, dtype=torch.float32)
            error, _ = masked_squared_error_sum(adapted_prompted.float(), target, video_mask)
            energy, _ = masked_squared_error_sum(target, torch.zeros_like(target), video_mask)
            if float(energy) > 0.0:
                self._velocity_errors.setdefault(sigma_bin.index, []).append(float(error) / float(energy))
            # The same error carried by the untouched checkpoint, and the ratio between
            # them.
            #
            # The absolute number is unreadable on its own: 0.10 is a good fit on one
            # dataset and a bad one on another, because it is measured against whatever
            # velocities that data happens to contain. The ratio is not -- 1.0 means the
            # adapter predicts the data no better than the checkpoint it started from,
            # and 0 would mean perfectly. That turns "did this run learn anything" into
            # a number readable from its own log, with no second run to compare against.
            #
            # It matters because an adapter can look excellent on every preservation
            # metric for the trivial reason that it barely moved. Measured: an arm that
            # scored best of its group on how little it disturbed the base had covered
            # about a seventh of the distance ordinary training covers, and the two facts
            # are the same fact. Reported beside the preservation numbers so that reading
            # cannot be made by accident.
            #
            # Free: the base prediction is already cached for the field comparison, so
            # this costs one more masked reduction and no forward.
            if base_pair is not None:
                base_error, _ = masked_squared_error_sum(base_pair[0].to(target.device).float(), target, video_mask)
                if float(base_error) > 0.0:
                    self._velocity_error_ratios.setdefault(sigma_bin.index, []).append(float(error) / float(base_error))

        base_gap = self._field_base_gaps[key]
        if base_pair is not None and adapted_pair is not None:
            prompted_drift = self._masked_rms(adapted_pair[0] - base_pair[0], video_mask)
            self._branch_drift.setdefault(sigma_bin.index, []).append(
                (prompted_drift, self._masked_rms(adapted_pair[1] - base_pair[1], video_mask))
            )
            # The same drift as a share of the base's own field, which is the form that
            # can be compared across datasets and across runs -- the raw value is in the
            # units of whatever velocities the data happened to contain.
            #
            # This is the number to read when asking whether an adapter still answers
            # prompts the way the checkpoint did. It is the only one here that no
            # preservation term can flatter: the empty branch cancels out of it exactly,
            # since (prompted_adapter - base_empty) - (base_prompted - base_empty) is
            # just prompted_adapter - base_prompted. A method that holds the empty
            # branch still scores nothing here for doing so.
            if base_gap > 0.0:
                self._prompted_drift_ratios.setdefault(sigma_bin.index, []).append(prompted_drift / base_gap)
        if base_gap <= 0.0:
            # The base has no field to lose at this state, so a ratio would divide by
            # noise. Skipping keeps one degenerate item from dominating the average.
            return
        if adapted_pair is None:
            return
        ratio = self._masked_rms(adapted_pair[0] - adapted_pair[1], video_mask) / base_gap
        self._field_ratios.setdefault(sigma_bin.index, []).append(ratio)
        if base_pair is not None:
            cosine = self._masked_cosine(adapted_pair[0] - adapted_pair[1], base_pair[0] - base_pair[1], video_mask)
            self._field_cosines.setdefault(sigma_bin.index, []).append(cosine)
            # The two halves folded into the one number that orders arms correctly.
            #
            # Length and angle can each be flattered by an adapter that ruins the other,
            # and reading either alone inverts the ranking. Measured on real runs: the
            # arm carrying the LONGEST field of a dozen -- 0.74 of the base where the
            # others sat near 0.55 -- was the FURTHEST from the base's field, because it
            # had bought that length by turning 45 degrees. By the ratio it led the
            # table; by this number it came last.
            #
            # ||F_arm - F_base|| / ||F_base|| = sqrt(1 + r^2 - 2 r cos), with r the ratio
            # already reported. Zero means the field was left exactly where it was.
            # Per item rather than from the pooled ratio and cosine, since the average of
            # the distances is not the distance between the averages.
            self._field_distances.setdefault(sigma_bin.index, []).append(
                math.sqrt(max(0.0, 1.0 + ratio * ratio - 2.0 * ratio * cosine))
            )
            self._null_field_ratios.setdefault(sigma_bin.index, []).append(
                self._masked_rms(adapted_pair[0] - base_pair[1], video_mask) / base_gap
            )

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        if getattr(args, "h3_validation_field_probe", False) and not getattr(args, "validation_dataset_config", None):
            # The probe reports a ratio measured on held-out items. Run on the training
            # set it would report how well the adapter reproduces the field where it
            # was fitted, which is the one place the number cannot be trusted.
            raise ValueError("--h3_validation_field_probe requires --validation_dataset_config")
        profile_steps = int(getattr(args, "h3_profile_steps", 0) or 0)
        if profile_steps < 0:
            raise ValueError("--h3_profile_steps must be non-negative")
        if profile_steps > 0 and getattr(args, "max_train_epochs", None) is None:
            self._check_h3_profile_window(profile_steps, args.max_train_steps)
        if args.h3_swiglu_chunk_rows < 0:
            raise ValueError("--h3_swiglu_chunk_rows must be non-negative")
        if args.h3_swiglu_chunk_rows and args.compile:
            raise ValueError("--h3_swiglu_chunk_rows is not supported with --compile")
        if args.h3_lora_token_refiner:
            if not args.network_module.endswith("lora_minimax_h3"):
                raise ValueError("--h3_lora_token_refiner requires --network_module networks.lora_minimax_h3")
            network_args = list(args.network_args or [])
            if any(value.startswith("h3_lora_token_refiner=") for value in network_args):
                raise ValueError(
                    "set H3 token-refiner targeting with --h3_lora_token_refiner, not a duplicate --network_args value"
                )
            network_args.append("h3_lora_token_refiner=true")
            args.network_args = network_args
        selected_targets = _parse_h3_lora_targets(args.h3_lora_targets)
        if args.h3_lora_token_refiner and selected_targets and "token_refiner" not in selected_targets:
            selected_targets["token_refiner"] = None
        if selected_targets:
            if not args.network_module.endswith("lora_minimax_h3"):
                raise ValueError("--h3_lora_targets requires --network_module networks.lora_minimax_h3")
            network_args = list(args.network_args or [])
            if any(
                value.startswith(
                    (
                        "include_patterns=",
                        "exclude_patterns=",
                        "h3_target_modules=",
                        "h3_target_blocks=",
                        "h3_attention_blocks=",
                        "h3_mlp_blocks=",
                    )
                )
                for value in network_args
            ):
                raise ValueError(
                    "--h3_lora_targets cannot be combined with include_patterns/exclude_patterns or duplicate H3 target arguments"
                )
            network_args.append("h3_target_modules=" + ",".join(selected_targets))
            attention_blocks = selected_targets.get("attention")
            mlp_blocks = selected_targets.get("mlp")
            if attention_blocks is not None:
                network_args.append("h3_attention_blocks=" + ",".join(str(index) for index in attention_blocks))
            if mlp_blocks is not None:
                network_args.append("h3_mlp_blocks=" + ",".join(str(index) for index in mlp_blocks))
            args.network_args = network_args
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0
        self.vae_frame_stride = 17
        self._crepa_config = parse_crepa_config(args.crepa)
        args.h3_load_dino_features = self._crepa_config is not None and self._crepa_config.mode == "dino"
        args.h3_dino_model = self._crepa_config.dino_model if args.h3_load_dino_features else None
        if args.validation_dataset_config:
            validation_config = config_utils.load_user_config(args.validation_dataset_config)
            general_batch_size = int(validation_config.get("general", {}).get("batch_size", 1))
            validation_batch_sizes = [
                int(dataset.get("batch_size", general_batch_size)) for dataset in validation_config.get("datasets", [])
            ] or [general_batch_size]
            if any(batch_size != 1 for batch_size in validation_batch_sizes):
                raise ValueError("MiniMax H3 validation requires batch_size = 1 in --validation_dataset_config")

        # H3 owns its own flow shifts because video and audio ride different
        # schedules (12 and 3) off one shared unshifted coordinate. The common
        # sampler must therefore hand us that coordinate *unshifted*: applying
        # --discrete_flow_shift as well would shift video twice and leave audio
        # on a schedule the model was never trained for.
        if not math.isclose(args.discrete_flow_shift, 1.0):
            raise ValueError(
                "MiniMax H3 requires --discrete_flow_shift 1.0; set the per-modality shifts with "
                "--h3_shift_video / --h3_shift_audio instead (defaults 12.0 / 3.0)"
            )
        if args.timestep_sampling not in _H3_BASE_TIMESTEP_SAMPLING:
            raise ValueError(
                f"MiniMax H3 --timestep_sampling {args.timestep_sampling!r} applies a model-specific or "
                "resolution-dependent shift before H3's own video/audio shifts. Use uniform (recommended), "
                "sigmoid, shift, logsnr, or sigma."
            )
        if args.num_timestep_buckets is not None and args.timestep_sampling == "sigma":
            raise ValueError(
                "MiniMax H3 --num_timestep_buckets is not consumed by --timestep_sampling sigma; "
                "use the recommended --timestep_sampling uniform or disable bucketing"
            )
        focus_probability = float(args.h3_timestep_focus_probability)
        if not 0.0 <= focus_probability <= 1.0:
            raise ValueError("--h3_timestep_focus_probability must lie in [0, 1]")
        if focus_probability > 0.0:
            if args.timestep_sampling != "uniform":
                raise ValueError("--h3_timestep_focus_probability requires --timestep_sampling uniform")
            if not 0.0 <= args.h3_timestep_focus_min < args.h3_timestep_focus_max <= 1.0:
                raise ValueError("H3 timestep focus bounds must satisfy 0 <= min < max <= 1")
            if args.min_timestep is not None or args.max_timestep is not None:
                raise ValueError("H3 timestep focus cannot be combined with --min_timestep or --max_timestep")
        for name in ("h3_shift_video", "h3_shift_audio"):
            value = float(getattr(args, name))
            if not 0.01 <= value <= 100.0:
                raise ValueError(f"--{name} must be in [0.01, 100.0], got {value}")
        if args.h3_image_flow_shift is not None and args.h3_image_flow_shift <= 0:
            raise ValueError("MiniMax H3 --h3_image_flow_shift must be positive when specified")
        modality_loss_weights = {
            "h3_video_loss_weight": float(args.h3_video_loss_weight),
            "h3_audio_loss_weight": float(args.h3_audio_loss_weight),
        }
        for name, value in modality_loss_weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"--{name} must be finite and non-negative")
        if not any(value > 0 for value in modality_loss_weights.values()):
            raise ValueError("at least one of --h3_video_loss_weight or --h3_audio_loss_weight must be positive")
        if args.h3_observed_modality == "video" and modality_loss_weights["h3_audio_loss_weight"] == 0:
            raise ValueError("--h3_observed_modality video trains audio and therefore requires --h3_audio_loss_weight > 0")
        if args.h3_observed_modality == "audio" and modality_loss_weights["h3_video_loss_weight"] == 0:
            raise ValueError("--h3_observed_modality audio trains video and therefore requires --h3_video_loss_weight > 0")
        self._guidance_scale_range = _parse_guidance_scale_range(getattr(args, "h3_guidance_scale_range", None))
        if self._guidance_scale_range is not None:
            if args.h3_guidance_distillation_scale is not None:
                raise ValueError(
                    "--h3_guidance_scale_range replaces the single --h3_guidance_distillation_scale; set one, not both"
                )
            # Every gate downstream asks whether a distillation scale is
            # configured. The range answers yes, and its midpoint is the point
            # value the drawn family is centred on, so the two-pass machinery,
            # its cache requirements and its metadata all read a meaningful
            # number without a second flag threaded through them.
            args.h3_guidance_distillation_scale = 0.5 * sum(self._guidance_scale_range)
            logger.info(
                "MiniMax H3 guidance scale drawn per sample in [%s, %s]",
                self._guidance_scale_range[0],
                self._guidance_scale_range[1],
            )
        if args.h3_guidance_distillation_scale is not None and args.h3_guidance_distillation_scale <= 1.0:
            raise ValueError("--h3_guidance_distillation_scale must be greater than 1, or omitted for one-pass training")
        overlay_weights = getattr(args, "h3_overlay_weights", None)
        overlay_multiplier = float(getattr(args, "h3_overlay_weights_multiplier", 1.0))
        if not math.isfinite(overlay_multiplier):
            raise ValueError("--h3_overlay_weights_multiplier must be finite")
        if overlay_multiplier != 1.0 and not overlay_weights:
            raise ValueError("--h3_overlay_weights_multiplier requires --h3_overlay_weights")
        if overlay_weights and not Path(overlay_weights).is_file():
            raise FileNotFoundError(f"--h3_overlay_weights file not found: {overlay_weights}")
        if args.h3_guidance_loss_form == "contrastive" and args.h3_guidance_distillation_scale is None:
            raise ValueError("--h3_guidance_loss_form contrastive requires --h3_guidance_distillation_scale")
        self._validate_rollout_args(args)
        if args.h3_guidance_null_source != "live" and args.h3_guidance_distillation_scale is None:
            raise ValueError("--h3_guidance_null_source requires --h3_guidance_distillation_scale")
        if args.h3_fuse_frozen_teachers and (
            args.h3_guidance_distillation_scale is None
            or args.h3_guidance_null_source != "frozen"
            or args.h3_base_preservation_loss_weight <= 0
        ):
            raise ValueError(
                "--h3_fuse_frozen_teachers requires guidance distillation, "
                "--h3_guidance_null_source frozen, and --h3_base_preservation_loss_weight > 0"
            )
        if args.h3_guidance_cfg_zero and args.h3_guidance_distillation_scale is None:
            raise ValueError("--h3_guidance_cfg_zero requires --h3_guidance_distillation_scale")
        if not math.isfinite(args.h3_guidance_distillation_probability) or not 0 < args.h3_guidance_distillation_probability <= 1:
            raise ValueError("--h3_guidance_distillation_probability must be finite and lie in (0, 1]")
        if args.h3_guidance_distillation_probability < 1.0 and args.h3_guidance_distillation_scale is None:
            raise ValueError("--h3_guidance_distillation_probability requires --h3_guidance_distillation_scale")
        if not math.isfinite(args.h3_base_preservation_loss_weight) or args.h3_base_preservation_loss_weight < 0:
            raise ValueError("--h3_base_preservation_loss_weight must be finite and non-negative")
        anchor_weight = float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0)
        if getattr(args, "h3_adapter_prompt_only", False):
            if anchor_weight > 0:
                raise ValueError(
                    "--h3_adapter_prompt_only makes the student's empty branch the frozen one by construction, so "
                    "--h3_guidance_null_anchor_weight would hold a difference that is identically zero at the cost "
                    "of two forwards a step; drop one of the two"
                )
            if args.h3_caption_dropout_rate > 0:
                raise ValueError(
                    "--h3_adapter_prompt_only switches the adapter off on empty-prompt forwards, so a caption-dropout "
                    "step would train nothing; set --h3_caption_dropout_rate 0"
                )
        audio_scale = getattr(args, "h3_guidance_audio_scale", None)
        if audio_scale is not None and (not math.isfinite(audio_scale) or audio_scale < 1.0):
            raise ValueError("--h3_guidance_audio_scale must be finite and at least 1")
        scale_sigma_max = float(getattr(args, "h3_guidance_scale_sigma_max", 1.0))
        if not math.isfinite(scale_sigma_max) or not 0 < scale_sigma_max <= 1:
            raise ValueError("--h3_guidance_scale_sigma_max must be finite and lie in (0, 1]")
        self._validation_multipliers = []
        raw_multipliers = str(getattr(args, "h3_validation_multipliers", "") or "").strip()
        if raw_multipliers:
            try:
                self._validation_multipliers = [float(item) for item in raw_multipliers.split(",") if item.strip()]
            except ValueError as error:
                raise ValueError("--h3_validation_multipliers must be comma-separated numbers") from error
            if any(not math.isfinite(m) or m < 0 for m in self._validation_multipliers):
                raise ValueError("--h3_validation_multipliers must be finite and non-negative")
        ema_decay = float(getattr(args, "h3_adapter_ema_decay", 0.0) or 0.0)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("--h3_adapter_ema_decay must be finite and lie in [0, 1); 0 disables it")
        if getattr(args, "h3_validate_ema", False) and ema_decay <= 0:
            raise ValueError("--h3_validate_ema requires --h3_adapter_ema_decay above 0")
        self._adapter_ema = None
        self._measured_variance_curve = None
        curve_path = getattr(args, "h3_measured_variance_weighting", None)
        weight_max = float(getattr(args, "h3_measured_variance_weight_max", 4.0))
        if not math.isfinite(weight_max) or weight_max < 1.0:
            raise ValueError("--h3_measured_variance_weight_max must be finite and at least 1")
        if curve_path:
            if not Path(curve_path).is_file():
                raise FileNotFoundError(f"--h3_measured_variance_weighting file not found: {curve_path}")
            self._measured_variance_curve = load_measured_variance_curve(curve_path, weight_max)
        if not math.isfinite(anchor_weight) or anchor_weight < 0:
            # Every gate on this feature reads "> 0", so a negative weight would
            # configure the anchor and then quietly train without it.
            raise ValueError("--h3_guidance_null_anchor_weight must be finite and non-negative; 0 disables it")
        anchor_probability = _anchor_probability(args)
        if not math.isfinite(anchor_probability) or not 0 < anchor_probability <= 1:
            raise ValueError("--h3_guidance_null_anchor_probability must be finite and lie in (0, 1]")
        if anchor_probability < 1.0 and anchor_weight <= 0:
            raise ValueError("--h3_guidance_null_anchor_probability requires --h3_guidance_null_anchor_weight above 0")
        if not math.isfinite(args.h3_base_preservation_probability) or not 0 < args.h3_base_preservation_probability <= 1:
            raise ValueError("--h3_base_preservation_probability must be finite and lie in (0, 1]")
        if not math.isfinite(args.h3_dop_loss_weight) or args.h3_dop_loss_weight < 0:
            raise ValueError("--h3_dop_loss_weight must be finite and non-negative")
        if not math.isfinite(args.h3_dop_probability) or not 0 < args.h3_dop_probability <= 1:
            raise ValueError("--h3_dop_probability must be finite and lie in (0, 1]")
        if args.h3_dop_loss_weight > 0:
            if not args.h3_dop_trigger or not args.h3_dop_class_prompt:
                raise ValueError("--h3_dop_loss_weight requires --h3_dop_trigger and --h3_dop_class_prompt")
            if args.h3_training_mode in {"ref2va", "ref2va_omni"}:
                raise ValueError("H3 DOP is not supported for Ref2VA")
        if args.h3_convrot_int8 and (args.fp8_base or args.int8_convrot_base):
            raise ValueError("--h3_convrot_int8 quantizes the BF16 checkpoint itself; drop --fp8_base/--int8_convrot_base")
        convrot_int8_active = args.h3_convrot_int8 or args.int8_convrot_base
        if args.h3_convrot_int8_bwd == "int8" and not convrot_int8_active:
            raise ValueError("--h3_convrot_int8_bwd int8 requires --h3_convrot_int8 or --int8_convrot_base")
        if args.h3_convrot_int8_fwd == "bf16" and not convrot_int8_active:
            raise ValueError("--h3_convrot_int8_fwd bf16 requires --h3_convrot_int8 or --int8_convrot_base")
        if args.h3_convrot_int8_fwd == "bf16" and args.h3_convrot_int8_bwd == "int8":
            raise ValueError("--h3_convrot_int8_fwd bf16 leaves no rotated activations for --h3_convrot_int8_bwd int8")
        if args.h3_convrot_int8_lora_fused and not (
            convrot_int8_active and args.h3_convrot_int8_fwd == "int8" and args.h3_convrot_int8_bwd == "int8"
        ):
            raise ValueError(
                "--h3_convrot_int8_lora_fused requires online or pre-quantized ConvRot INT8 weights with "
                "--h3_convrot_int8_fwd int8 and --h3_convrot_int8_bwd int8"
            )
        if convrot_int8_active and args.block_swap_granularity == "layer":
            raise ValueError(
                "--block_swap_granularity layer bypasses the ConvRot INT8 forward and corrupts the base output; "
                "use --block_swap_granularity block or drop --h3_convrot_int8/--int8_convrot_base"
            )
        if not 0.0 <= args.h3_caption_dropout_rate <= 1.0:
            raise ValueError("--h3_caption_dropout_rate must lie in [0, 1]")
        if not math.isfinite(args.h3_qwen_control_dropout_rate) or not 0.0 <= args.h3_qwen_control_dropout_rate <= 1.0:
            raise ValueError("--h3_qwen_control_dropout_rate must be finite and lie in [0, 1]")
        if args.h3_extension_video_frames < 0 or args.h3_extension_audio_latents < 0:
            raise ValueError("H3 extension context lengths must be non-negative")
        self._extension_video_frames = args.h3_extension_video_frames
        self._extension_audio_latents = args.h3_extension_audio_latents
        self._extension_route = args.h3_extension_route
        self._block_swap_h2d_only = bool(args.block_swap_h2d_only)
        self._frame_sigma_jitter = args.h3_frame_sigma_jitter
        if not 0.0 <= args.h3_frame_sigma_jitter <= 1.0:
            raise ValueError("--h3_frame_sigma_jitter must lie in [0, 1]")
        self._spatial_density_jitter = args.h3_spatial_density_jitter
        if not math.isfinite(args.h3_spatial_density_jitter) or args.h3_spatial_density_jitter < 0:
            raise ValueError("--h3_spatial_density_jitter must be finite and non-negative")
        self._keyframe_anchors = _parse_keyframe_anchors(args.h3_keyframe_anchors)
        self._keyframe_random_count = args.h3_keyframe_random_count
        self._guide_specs = _parse_guide_specs(getattr(args, "h3_guide_specs", ""))
        if args.h3_keyframe_random_count < 0:
            raise ValueError("--h3_keyframe_random_count cannot be negative")
        if self._keyframe_anchors and args.h3_keyframe_random_count:
            raise ValueError("H3 keyframe anchors are either listed or drawn at random, not both")
        keyframes = bool(self._keyframe_anchors) or bool(args.h3_keyframe_random_count) or bool(self._guide_specs)
        if keyframes and (args.h3_extension_video_frames or args.h3_extension_audio_latents):
            raise ValueError("H3 keyframe conditioning and extension both claim the observed rows; enable only one")
        if keyframes and (args.h3_mask_mode != "off" or args.h3_mask_audio):
            raise ValueError("H3 keyframe conditioning and masked conditioning both claim the observed rows; enable only one")
        if keyframes and args.h3_training_mode not in ("fl2va", "ref2va", "ref2va_omni"):
            raise ValueError("H3 keyframe conditioning requires FL2VA or Ref2VA training with video targets")
        if self._guide_specs and args.h3_training_mode == "fl2va":
            raise ValueError("--h3_guide_specs currently requires Ref2VA or Ref2VA-Omni training")
        self._mask_mode = args.h3_mask_mode
        self._mask_audio = args.h3_mask_audio
        self._mask_bounds = (args.h3_mask_min_fraction, args.h3_mask_max_fraction)
        if not 0.0 < args.h3_mask_min_fraction <= args.h3_mask_max_fraction <= 1.0:
            raise ValueError("H3 mask fractions must satisfy 0 < min <= max <= 1")
        masking = args.h3_mask_mode != "off" or args.h3_mask_audio
        extension = bool(args.h3_extension_video_frames or args.h3_extension_audio_latents)
        for name in ("h3_mask_probability", "h3_extension_probability"):
            value = float(getattr(args, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be finite and lie in [0, 1]")
        if args.h3_mask_probability < 1.0 and not masking:
            raise ValueError("--h3_mask_probability requires --h3_mask_mode or --h3_mask_audio")
        if args.h3_extension_probability < 1.0 and not extension:
            raise ValueError("--h3_extension_probability requires --h3_extension_video_frames or --h3_extension_audio_latents")
        if masking and extension:
            # Both recipes claim the observed rows, so they may share a run only
            # when a per-step draw picks at most one of them for each step.
            if args.h3_mask_probability >= 1.0 and args.h3_extension_probability >= 1.0:
                raise ValueError("H3 masked conditioning and extension both claim the observed rows; enable only one")
            if args.h3_mask_probability + args.h3_extension_probability > 1.0:
                raise ValueError(
                    "--h3_mask_probability and --h3_extension_probability select at most one recipe per step, "
                    "so together they must not exceed 1"
                )
        # Masking and per-row-sigma extension only pin rows inside the target
        # block, which every layout carries, so both combine with Ref2VA.
        # condition_rows extension instead duplicates the observed span as extra
        # clean rows, which only the T2VA packer knows how to place.
        if (
            (args.h3_extension_video_frames or args.h3_extension_audio_latents)
            and args.h3_training_mode != "fl2va"
            and args.h3_extension_route != "per_row_sigma"
        ):
            raise ValueError(
                f"H3 extension under --h3_training_mode {args.h3_training_mode} requires "
                "--h3_extension_route per_row_sigma; --h3_extension_route condition_rows needs "
                "--h3_training_mode fl2va with --task t2va caches"
            )
        # Jitter re-noises the whole video at per-frame levels, which silently
        # overwrites any row a conditioning mode pinned as observed and leaves
        # the row timesteps disagreeing with the noise actually applied.
        conditioning = (
            keyframes
            or masking
            or bool(args.h3_extension_video_frames or args.h3_extension_audio_latents)
            or args.h3_observed_modality is not None
        )
        if args.h3_frame_sigma_jitter > 0 and conditioning:
            raise ValueError(
                "--h3_frame_sigma_jitter re-noises every frame, so it cannot be combined with a conditioning mode "
                "that presents part of the target as observed (--h3_observed_modality, extension, keyframes, or masking)"
            )
        if args.h3_frame_sigma_jitter > 0 and args.weighting_scheme in {"sigma_sqrt", "cosmap"}:
            raise ValueError(
                f"--h3_frame_sigma_jitter cannot be combined with --weighting_scheme {args.weighting_scheme}: "
                "per-frame weighting is not supported"
            )
        if args.h3_sigma_sqrt_max_weight <= 0:
            raise ValueError("MiniMax H3 --h3_sigma_sqrt_max_weight must be positive")
        if args.reference_image_max_pixels < 0:
            raise ValueError("MiniMax H3 --reference_image_max_pixels must be non-negative")
        # Defer to the canonical validator so a value accepted here cannot fail
        # later inside reference_key_suffix() with a different lower bound.
        validate_reference_video_sizing(args.reference_video_short_edge, args.reference_video_max_pixels)
        validate_reference_video_fps(args.reference_video_fps)
        if args.h3_max_caption_tokens < 0:
            raise ValueError("MiniMax H3 --h3_max_caption_tokens must be non-negative")
        if args.reference_image_size_mode == "short_edge" and args.reference_image_max_pixels:
            raise ValueError("--reference_image_max_pixels applies only to --reference_image_size_mode target_area")
        if args.h3_guidance_distillation_scale is not None and float(getattr(args, "network_dropout", 0.0) or 0.0) > 0:
            raise ValueError(
                "H3 guidance-consistent training cannot replay --network_dropout across different prompt lengths; "
                "use rank_dropout or module_dropout instead"
            )
        if args.fp8_base and args.h3_adaln_rank is None:
            # AdaLN is ~39% of the transformer and is quantized by default, yet
            # measured against the BF16 reference the reduction is both smaller
            # and more faithful than quantizing it.
            logger.info(
                "MiniMax H3: --fp8_base quantizes the AdaLN projections. Reducing them instead with "
                "--h3_adaln_rank 16 is both smaller and closer to the BF16 reference; consider adding it."
            )
        if args.fp8_base:
            # H3 supports only weight-only scaled FP8. Reuse the common
            # --fp8_base switch without exposing an H3-only parser field, and
            # prevent the base trainer from casting the mixed-precision shell
            # and norms directly to float8.
            args.fp8_scaled = True
        if args.int8_convrot_base and args.fp8_base:
            raise ValueError("MiniMax H3 --int8_convrot_base cannot be combined with --fp8_base")
        if args.blocks_to_swap is not None and args.blocks_to_swap < 0:
            raise ValueError("MiniMax H3 --blocks_to_swap must be non-negative")
        if args.h3_gradient_checkpointing_blocks is not None:
            checkpoint_blocks = args.h3_gradient_checkpointing_blocks
            if not 0 <= checkpoint_blocks <= 50:
                raise ValueError("--h3_gradient_checkpointing_blocks must be in [0, 50]")
            if not args.gradient_checkpointing:
                raise ValueError("--h3_gradient_checkpointing_blocks requires --gradient_checkpointing")
            if checkpoint_blocks < 50 and (args.blocks_to_swap or 0) > 0:
                # Every swap implementation streams a block's weights through a
                # buffer that is repointed or overwritten in place once the block's
                # forward has been consumed. Only checkpoint recomputation re-reads
                # those weights at backward time; an eager block instead saves the
                # streamed view directly into the autograd graph, so backward reads
                # either a stale ring slot (h2d_only, which is why block swap
                # requires gradient checkpointing at all) or a CPU-resident storage.
                raise ValueError("partial H3 gradient checkpointing cannot be combined with block swap")
        checkpoint_keep = getattr(args, "h3_checkpoint_keep", "none")
        if checkpoint_keep != "none":
            # The policy keeps activations in the selective checkpoint cache,
            # which neither the CPU offload hooks nor a compiled block can see,
            # and it only works where the attention forward is a dispatcher op.
            if not args.gradient_checkpointing:
                raise ValueError("--h3_checkpoint_keep requires --gradient_checkpointing")
            if args.gradient_checkpointing_cpu_offload:
                raise ValueError("--h3_checkpoint_keep cannot be combined with --gradient_checkpointing_cpu_offload")
            if args.compile:
                raise ValueError("--h3_checkpoint_keep cannot be combined with --compile")
            if getattr(args, "h3_int8_attention", "off") == "train":
                raise ValueError("--h3_checkpoint_keep cannot be combined with --h3_int8_attention train")
            if getattr(args, "h3_block_sparse_kv_fraction", 0.0) > 0 or getattr(args, "h3_block_sparse_threshold", 0.0) > 0:
                raise ValueError("--h3_checkpoint_keep cannot be combined with block-sparse attention")
            if (args.blocks_to_swap or 0) > 0 and getattr(args, "block_swap_granularity", "block") == "layer":
                raise ValueError("--h3_checkpoint_keep cannot be combined with --block_swap_granularity layer")
            if checkpoint_keep == "qkv" and args.h3_convrot_int8_lora_fused:
                # The fused kernel runs base and adapter in one Triton launch,
                # bypassing the base projection's forward the region wraps.
                raise ValueError("--h3_checkpoint_keep qkv cannot be combined with --h3_convrot_int8_lora_fused")
        if args.h3_gradient_checkpointing_cpu_offload_pin_memory and not (
            args.gradient_checkpointing and args.gradient_checkpointing_cpu_offload
        ):
            raise ValueError(
                "--h3_gradient_checkpointing_cpu_offload_pin_memory requires "
                "--gradient_checkpointing and --gradient_checkpointing_cpu_offload"
            )
        if args.h3_reusable_activation_offload and not (args.gradient_checkpointing and args.gradient_checkpointing_cpu_offload):
            raise ValueError(
                "--h3_reusable_activation_offload requires --gradient_checkpointing and --gradient_checkpointing_cpu_offload"
            )
        if getattr(args, "gradient_checkpointing_cpu_offload_dtype", "none") != "none":
            if not (args.gradient_checkpointing and args.gradient_checkpointing_cpu_offload):
                raise ValueError(
                    "--gradient_checkpointing_cpu_offload_dtype requires "
                    "--gradient_checkpointing and --gradient_checkpointing_cpu_offload"
                )
            if not args.h3_reusable_activation_offload:
                # Compression lives at the reusable offloader's pack/unpack
                # boundary; the plain ``save_on_cpu`` path has no hook to apply
                # it, so silently ignoring the flag there would be worse.
                raise ValueError("--gradient_checkpointing_cpu_offload_dtype requires --h3_reusable_activation_offload")
        if args.block_swap_h2d_only and not args.use_pinned_memory_for_block_swap:
            logger.warning(
                "MiniMax H3 H2D-only block swap without pinned host memory uses staged copies and can be substantially slower; "
                "add --use_pinned_memory_for_block_swap for direct asynchronous transfers"
            )
        if not (args.sdpa or args.flash_attn or args.flash3):
            raise ValueError("MiniMax H3 training requires --sdpa, --flash_attn, or --flash3")
        if args.h3_attn_auto_dispatch and not args.sdpa:
            raise ValueError("--h3_attn_auto_dispatch requires --sdpa")
        if getattr(args, "h3_int8_attention", "off") != "off" and args.compile:
            raise ValueError("--h3_int8_attention cannot currently be combined with --compile")
        if getattr(args, "h3_compile_attention", "inline") == "opaque" and getattr(args, "compile_fullgraph", False):
            raise ValueError("--h3_compile_attention opaque is one graph break per block, which --compile_fullgraph forbids")
        block_sparse_kv_fraction = getattr(args, "h3_block_sparse_kv_fraction", 0.0)
        block_sparse_threshold = getattr(args, "h3_block_sparse_threshold", 0.0)
        # The runtime config rejects the same bounds, but only on the first
        # forward, after model loading and caching have already been paid for.
        if not 0.0 <= block_sparse_kv_fraction <= 1.0:
            raise ValueError("--h3_block_sparse_kv_fraction must be in [0, 1]")
        if not 0.0 <= block_sparse_threshold <= 1.0:
            raise ValueError("--h3_block_sparse_threshold must be in [0, 1]")
        _parse_block_sparse_block_shape(getattr(args, "h3_block_sparse_block_shape", None))
        if block_sparse_kv_fraction > 0 or block_sparse_threshold > 0:
            if getattr(args, "h3_int8_attention", "off") != "off":
                # Block-sparse attention takes every unmasked call and masked
                # calls fall back to dense, so INT8 attention would never run.
                raise ValueError(
                    "--h3_block_sparse_kv_fraction/--h3_block_sparse_threshold cannot be combined with --h3_int8_attention"
                )
            if args.compile:
                # flex_attention compiles its own kernel; nesting that inside
                # region-compiled blocks is untested.
                raise ValueError(
                    "--h3_block_sparse_kv_fraction/--h3_block_sparse_threshold cannot currently be combined with --compile"
                )
        if args.split_attn:
            raise ValueError("MiniMax H3 training does not support split attention")
        if args.sample_prompts:
            if args.h3_training_mode != "fl2va":
                raise ValueError(
                    "MiniMax H3 training-time sampling currently supports only FL2VA; "
                    "use minimax_h3_generate_video.py for Ref2VA samples"
                )
            required = {
                "--text_encoder": args.text_encoder,
                "--vae": args.vae,
                "--audio_vae": args.audio_vae,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("MiniMax H3 sampling during training requires " + ", ".join(missing))

    @staticmethod
    def _check_h3_profile_window(profile_steps: int, max_train_steps: int) -> None:
        needed = H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + profile_steps
        if int(max_train_steps) < needed:
            raise ValueError(
                f"--h3_profile_steps {profile_steps} records optimizer steps "
                f"{H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + 1}..{needed}, but the run has only "
                f"{max_train_steps}; raise --max_train_steps or lower --h3_profile_steps"
            )

    def train(self, args):
        try:
            return super().train(args)
        finally:
            self._finalize_h3_profiler()

    def _finalize_h3_profiler(self) -> None:
        profiler, self._h3_profiler = self._h3_profiler, None
        if profiler is not None:
            profiler.abort()

    def on_train_start(self, args, accelerator, network, transformer, optimizer) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        if getattr(args, "h3_fused_elementwise", False) and isinstance(network, torch.nn.Module):
            # The adapters' delta add joins the single-rounding regime for factors that are not
            # powers of two (power-of-two factors already take the fused, bit-identical path).
            for module in network.modules():
                if hasattr(module, "fused_scale_add"):
                    module.fused_scale_add = True
        self._h3_profiler = None
        steps = int(getattr(args, "h3_profile_steps", 0) or 0)
        if steps <= 0:
            return
        # max_train_steps is final here (an epoch count has been converted by now).
        self._check_h3_profile_window(steps, args.max_train_steps)
        if not accelerator.is_main_process:
            return
        output_name = getattr(args, "output_name", None) or "h3"
        output_path = Path(args.output_dir) / f"{output_name}_profile.txt"
        self._h3_profiler = H3StepProfiler(steps, accelerator.device, output_path)
        self._h3_profiler.start()

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step) -> None:
        super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)
        if sync_gradients:
            self._update_adapter_ema(args, accelerator, network, global_step)
        if self._h3_profiler is None or not sync_gradients:
            return
        self._h3_profiler.step()
        if self._h3_profiler.finished:
            self._h3_profiler = None

    # ------------------------------------------------------------------ adapter EMA
    @staticmethod
    def _adapter_parameters(accelerator, network) -> list[tuple[str, torch.nn.Parameter]]:
        unwrapped = accelerator.unwrap_model(network)
        return [(name, parameter) for name, parameter in unwrapped.named_parameters() if parameter.requires_grad]

    def _update_adapter_ema(self, args, accelerator, network, global_step: int) -> None:
        """Track the trainable parameters' EMA and save it beside each scheduled checkpoint.

        Called after every optimizer step (the base loop increments its step
        counter afterwards, so the step this update belongs to is ``global_step
        + 1``). The average is kept in fp32 on the parameters' device; LoRA
        parameters are small enough for that.
        """
        decay = float(getattr(args, "h3_adapter_ema_decay", 0.0) or 0.0)
        if decay <= 0 or network is None:
            return
        parameters = self._adapter_parameters(accelerator, network)
        with torch.no_grad():
            if self._adapter_ema is None:
                self._adapter_ema = {name: parameter.detach().float().clone() for name, parameter in parameters}
            else:
                for name, parameter in parameters:
                    shadow = self._adapter_ema[name]
                    shadow.mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)
        step = int(global_step) + 1
        every = getattr(args, "save_every_n_steps", None)
        if every and step % int(every) == 0 and accelerator.is_main_process:
            self._save_adapter_ema(args, accelerator, network, step)

    @contextmanager
    def _adapter_ema_weights(self, accelerator, network):
        """Swap the EMA into the trainable parameters for the duration of the block."""
        if self._adapter_ema is None or network is None:
            yield False
            return
        parameters = self._adapter_parameters(accelerator, network)
        backup = {}
        with torch.no_grad():
            for name, parameter in parameters:
                backup[name] = parameter.detach().clone()
                parameter.copy_(self._adapter_ema[name].to(dtype=parameter.dtype))
        try:
            yield True
        finally:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(backup[name])

    def _save_adapter_ema(self, args, accelerator, network, step: int) -> None:
        unwrapped = accelerator.unwrap_model(network)
        save_weights = getattr(unwrapped, "save_weights", None)
        if not callable(save_weights):
            logger.warning("--h3_adapter_ema_decay: the network has no save_weights(); the EMA is kept but not saved")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, f"{args.output_name}-ema-step{step:08d}.safetensors")
        dtype = model_utils.str_to_dtype(getattr(args, "save_precision", None), torch.bfloat16)
        metadata = dict(self.extra_metadata(args))
        metadata["ss_h3_adapter_ema"] = "True"
        metadata["ss_steps"] = str(step)
        with self._adapter_ema_weights(accelerator, network):
            save_weights(path, dtype, metadata)
        logger.info("saved adapter EMA: %s", path)

    def on_transformer_loaded(self, args, accelerator, transformer) -> None:
        transformer.set_gradient_checkpointing_blocks(args.h3_gradient_checkpointing_blocks)
        transformer.set_activation_cpu_offload_pin_memory(args.h3_gradient_checkpointing_cpu_offload_pin_memory)
        checkpoint_keep = getattr(args, "h3_checkpoint_keep", "none")
        if checkpoint_keep != "none":
            transformer.set_checkpoint_keep(checkpoint_keep)
        fraction = getattr(args, "h3_block_sparse_kv_fraction", 0.0)
        threshold = getattr(args, "h3_block_sparse_threshold", 0.0)
        set_block_sparse = getattr(transformer, "set_block_sparse_attention", None)
        if (fraction > 0 or threshold > 0) and callable(set_block_sparse):
            from musubi_tuner.minimax_h3.block_sparse_attention import BlockSparseConfig

            set_block_sparse(
                BlockSparseConfig(
                    kv_fraction=fraction if fraction > 0 else 1.0,
                    threshold=threshold if threshold > 0 else None,
                    block_shape=_parse_block_sparse_block_shape(getattr(args, "h3_block_sparse_block_shape", None)),
                ),
                start_block=getattr(args, "h3_block_sparse_start_block", 0),
            )
        set_int8_attention_mode = getattr(transformer, "set_int8_attention_mode", None)
        if callable(set_int8_attention_mode):
            set_int8_attention_mode(getattr(args, "h3_int8_attention", "off"))
        if args.h3_reusable_activation_offload:
            transformer.enable_reusable_activation_offload()
            offload_dtype = getattr(args, "gradient_checkpointing_cpu_offload_dtype", "none")
            if offload_dtype != "none":
                transformer.reusable_activation_offloader.set_offload_dtype(offload_dtype)
        if args.h3_fused_qk_norm_rope:
            transformer.enable_fused_qk_norm_rope()
            if args.compile:
                logger.info(
                    "--h3_fused_qk_norm_rope requested with --compile: compiled blocks use Inductor fusion; "
                    "the explicit Triton kernel remains active for eager calls"
                )
        if getattr(args, "h3_fused_indexed_adaln", False):
            transformer.enable_fused_indexed_adaln()
            if args.compile:
                logger.warning("--h3_fused_indexed_adaln falls back to the Inductor path inside compiled blocks")
        if getattr(args, "h3_fused_swiglu", False):
            transformer.enable_fused_swiglu()
            if args.compile:
                logger.warning("--h3_fused_swiglu falls back to the Inductor path inside compiled blocks")
        if getattr(args, "h3_fused_elementwise", False):
            transformer.enable_fused_elementwise()
        set_swiglu_chunk_rows = getattr(transformer, "set_swiglu_chunk_rows", None)
        if callable(set_swiglu_chunk_rows):
            set_swiglu_chunk_rows(getattr(args, "h3_swiglu_chunk_rows", 0))
        if args.h3_convrot_int8_lora_fused:
            from musubi_tuner.modules.convrot_int8_utils import enable_convrot_int8_lora_fusion

            enabled = enable_convrot_int8_lora_fusion(transformer)
            if enabled == 0:
                raise RuntimeError("--h3_convrot_int8_lora_fused found no ConvRot INT8 Linear layers")

        sampler_state_path = "h3_timestep_sampler.json"

        def save_sampler_state(_models, _weights, output_dir):
            if accelerator.is_main_process:
                state = {
                    "num_timestep_buckets": self.num_timestep_buckets,
                    "timestep_range_pool": self.timestep_range_pool,
                }
                (Path(output_dir) / sampler_state_path).write_text(json.dumps(state), encoding="utf-8")

        def load_sampler_state(_models, input_dir):
            path = Path(input_dir) / sampler_state_path
            if not path.exists():
                # Older checkpoints did not save the partially consumed pool.
                # Starting a fresh cycle is safe, but cannot exactly reproduce
                # the pre-resume bucket ordering.
                self.timestep_range_pool = []
                return
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("num_timestep_buckets") != self.num_timestep_buckets:
                raise ValueError("saved H3 timestep sampler state does not match --num_timestep_buckets")
            pool = state.get("timestep_range_pool")
            if not isinstance(pool, list) or any(not isinstance(item, list) or len(item) != 2 for item in pool):
                raise ValueError(f"invalid H3 timestep sampler state: {path}")
            self.timestep_range_pool = [(float(lower), float(upper)) for lower, upper in pool]

        accelerator.register_save_state_pre_hook(save_sampler_state)
        accelerator.register_load_state_pre_hook(load_sampler_state)
        if self._crepa_config is None:
            return
        config = getattr(transformer, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise TypeError("MiniMax H3 CREPA requires transformer.config.hidden_size")
        self._crepa = H3CREPA(hidden_size, self._crepa_config)
        self._crepa.install(transformer)
        object.__setattr__(transformer, "_h3_crepa_controller", self._crepa)

        def save_crepa_state(_models, _weights, output_dir):
            if accelerator.is_main_process:
                self._crepa.save_state(output_dir)

        def load_crepa_state(_models, input_dir):
            if not self._crepa.load_state(input_dir):
                raise FileNotFoundError(f"CREPA resume state is missing: {Path(input_dir) / 'h3_crepa.safetensors'}")

        accelerator.register_save_state_pre_hook(save_crepa_state)
        accelerator.register_load_state_pre_hook(load_crepa_state)

    def install_overlay_weights(self, args, accelerator, transformer):
        """Attach ``--h3_overlay_weights`` as a live, frozen LoRA overlay.

        The overlay is a second :class:`LoRANetwork` applied on top of the
        trainable one. Nothing is written into the transformer's weights, which
        is the entire point: a ConvRot INT8 base cannot absorb a LoRA without a
        dequantize/requantize round trip, so the correction has to stay a
        separate live module. Because ``LoRAModule.apply_to`` chains onto
        whatever forward it finds, applying the overlay *after* the trainable
        network leaves the trainable modules innermost, so they keep the fused
        ConvRot INT8 LoRA path when it is enabled, and the two deltas simply
        add.

        Ownership follows from the object graph rather than from bookkeeping:
        the overlay is not a submodule of the trainable network, is not handed
        to the optimizer, and is not passed to the accelerator, so it cannot
        reach a checkpoint, a saved ``state_dict`` or a gradient. Likewise the
        base-preservation branch toggles ``set_enabled`` on the trainable
        network alone, which leaves the overlay active on the reference forward:
        the preserved field is base+overlay, the field the adapter is actually
        being trained inside.
        """
        path = getattr(args, "h3_overlay_weights", None)
        if not path:
            return None
        from musubi_tuner.networks import lora_minimax_h3

        weights_sd = self.load_network_weights(path, "musubi_tuner.networks.lora_minimax_h3")
        multiplier = float(getattr(args, "h3_overlay_weights_multiplier", 1.0))
        overlay = lora_minimax_h3.create_arch_network_from_weights(
            multiplier,
            weights_sd,
            unet=transformer,
            for_inference=False,
        )
        matched = {lora.lora_name for lora in overlay.text_encoder_loras + overlay.unet_loras}
        unmatched = sorted({key.split(".")[0] for key in weights_sd if "." in key} - matched)
        if unmatched:
            raise ValueError(
                f"{path}: {len(unmatched)} of {len(weights_sd)} tensors have no matching module and would be applied "
                "to nothing; a live overlay is never partially applied. Its key convention is probably not the one "
                f"this project expects (`lora_unet_<module_path_with_underscores>.lora_down.weight`), e.g. "
                f"{', '.join(unmatched[:3])}"
            )
        overlay.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        info = overlay.load_state_dict(weights_sd, False)
        if info.missing_keys:
            raise ValueError(f"{path}: overlay LoRA is missing {len(info.missing_keys)} tensor(s), e.g. {info.missing_keys[:3]}")
        # fp32 matches the trainable network, which stays fp32 and relies on
        # autocast for the matmul dtype, so both deltas are computed alike.
        overlay.to(device=accelerator.device, dtype=torch.float32)
        overlay.requires_grad_(False)
        overlay.eval()
        ranks = sorted({int(lora.lora_dim) for lora in overlay.unet_loras})
        alphas = sorted({float(lora.alpha.item()) for lora in overlay.unet_loras})
        accelerator.print(
            f"MiniMax H3 live overlay from {path}: {len(overlay.unet_loras)} module(s) matched, "
            f"rank={ranks if len(ranks) > 1 else ranks[0]}, alpha={alphas if len(alphas) > 1 else alphas[0]}, "
            f"multiplier={multiplier}; frozen, never merged, excluded from checkpoints"
        )
        self._overlay_network = overlay
        return overlay

    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        if args is not None and (args.h3_base_preservation_loss_weight > 0 or args.h3_dop_loss_weight > 0):
            if network is None:
                raise ValueError("H3 preservation losses require a trainable network")
            set_enabled = getattr(accelerator.unwrap_model(network), "set_enabled", None)
            if not callable(set_enabled):
                raise TypeError("H3 preservation losses require a network with set_enabled()")
        if args is not None:
            self.install_overlay_weights(args, accelerator, transformer)
        del args, network, transformer
        if self._crepa is None:
            return trainable_params
        self._crepa.projector.to(device=accelerator.device, dtype=torch.float32)
        if not trainable_params or not isinstance(trainable_params[0], dict) or "params" not in trainable_params[0]:
            raise TypeError("MiniMax H3 CREPA requires the network optimizer parameters to use named parameter groups")
        groups = [dict(group) for group in trainable_params]
        groups[0]["params"] = [*groups[0]["params"], *self._crepa.projector.parameters()]
        return groups

    def extra_gradient_params(self) -> list[torch.nn.Parameter]:
        return [] if self._crepa is None else list(self._crepa.projector.parameters())

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        prompts = load_prompts(sample_prompts)
        logger.info("Encoding %d MiniMax H3 sampling prompt(s)", len(prompts))
        encoder = create_conditioning_encoder(
            text_encoder=Path(args.text_encoder),
            tokenizer=Path(args.tokenizer),
            task="t2va",
            device=str(accelerator.device),
            dtype="bfloat16",
            quantization=args.text_encoder_quantization,
            blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
            nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
            text_visual_max_pixels=args.h3_text_visual_max_pixels,
        )
        prepared_images: list[list[Image.Image]] = []
        for prompt in prompts:
            height = prompt.get("height", 192)
            width = prompt.get("width", 320)
            images = []
            anchors = []
            if prompt.get("image_path"):
                with Image.open(prompt["image_path"]) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=True))
                anchors.append("first")
            if prompt.get("end_image_path"):
                with Image.open(prompt["end_image_path"]) as image:
                    images.append(prepare_keyframe_image(image, height, width, stretch=False))
                anchors.append("last")
            prompt.update(encoder.encode_prompt(prompt.get("prompt", ""), images))
            prompt[_SAMPLE_KEYFRAME_ANCHORS] = tuple(anchors)
            prepared_images.append(images)
        encoder.close()
        del encoder
        gc.collect()
        clean_memory_on_device(accelerator.device)
        all_images = [image for images in prepared_images for image in images]
        encoded = iter(encode_keyframe_images(Path(args.vae), all_images, accelerator.device))
        for prompt, images in zip(prompts, prepared_images):
            rows = [next(encoded) for _ in images]
            prompt[_SAMPLE_KEYFRAME_ROWS] = torch.cat(rows) if rows else None
        return prompts

    def _generate_sample(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        decoder_bundle: _H3DecoderBundle,
        sample_parameter: dict,
    ):
        device = accelerator.device
        height = sample_parameter.get("height", 192)
        width = sample_parameter.get("width", 320)
        frame_count = align_frame_count(sample_parameter.get("frame_count", 124))
        sample_steps = sample_parameter.get("sample_steps", 20)
        seed = sample_parameter.get("seed", 42)
        generator = torch.Generator(device=device).manual_seed(seed)
        conditioning = {
            H3_TEXT_HIDDEN_KEY: sample_parameter[H3_TEXT_HIDDEN_KEY],
            H3_TEXT_TOKEN_TAGS_KEY: sample_parameter[H3_TEXT_TOKEN_TAGS_KEY],
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        denoise_started = time.perf_counter()
        video_latents, audio_latents = denoise_fl2va(
            transformer,
            conditioning,
            height=height,
            width=width,
            frame_count=frame_count,
            num_inference_steps=sample_steps,
            generator=generator,
            device=device,
            keyframe_rows=sample_parameter.get(_SAMPLE_KEYFRAME_ROWS),
            keyframe_anchors=sample_parameter.get(_SAMPLE_KEYFRAME_ANCHORS, ()),
            condition_seed=seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sample_metrics = {
            "joint_denoising": {
                "seconds": time.perf_counter() - denoise_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            }
        }
        video_latents = video_latents.cpu()
        audio_latents = audio_latents.cpu()

        block_swap_suspended = bool(self.blocks_to_swap)
        if block_swap_suspended:
            transformer.offload_block_swap_to_cpu()
        else:
            transformer.to("cpu")
        clean_memory_on_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        decode_started = time.perf_counter()
        try:
            media = decode_latents_sequentially(
                decoder_bundle.video_decoder,
                decoder_bundle.audio_decoder,
                video_latents,
                audio_latents,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_metrics["sequential_av_decode"] = {
                "seconds": time.perf_counter() - decode_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            }
            self._last_sample_metrics = sample_metrics
            logger.info("MiniMax H3 training sample metrics: %s", sample_metrics)
        finally:
            if block_swap_suspended:
                transformer.move_to_device_except_swap_blocks(device)
                transformer.switch_block_swap_for_inference()
            else:
                transformer.to(device)
        return media, height, width, frame_count, sample_steps, seed

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
    ):
        del dit_dtype
        media, height, width, frame_count, sample_steps, seed = self._generate_sample(
            accelerator,
            transformer,
            vae,
            sample_parameter,
        )
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        checkpoint = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        prompt_index = sample_parameter.get("enum", 0)
        prefix = "" if args.output_name is None else args.output_name + "_"
        output = Path(save_dir) / f"{prefix}{checkpoint}_{prompt_index:02d}_{timestamp}_{seed}.mp4"
        save_av_mp4(
            media,
            output,
            {
                "training_step": steps,
                "epoch": epoch,
                "prompt": sample_parameter.get("prompt", ""),
                "seed": seed,
                "height": height,
                "width": width,
                "frames": frame_count,
                "sigma_points": sample_steps,
                "model_evaluations": sample_steps - 1,
                "keyframe_anchors": list(sample_parameter.get(_SAMPLE_KEYFRAME_ANCHORS, ())),
                "metrics": self._last_sample_metrics,
            },
        )
        logger.info("Saved MiniMax H3 AV sample to %s", output)

    def do_inference(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 sampling uses its AV-aware sample_image_inference implementation")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del vae_dtype, vae_path
        logger.info("Loading MiniMax H3 video/audio decoders on CPU for sampling")
        return _H3DecoderBundle(
            load_video_vae_decoder(Path(args.vae), "cpu"),
            load_audio_vae_decoder(Path(args.audio_vae), "cpu"),
        )

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        if args.fp8_base and dit_weight_dtype is not None:
            raise ValueError("MiniMax H3 scaled FP8 loading requires dit_weight_dtype=None")
        base_weight_paths = list(getattr(args, "base_weights", None) or [])
        base_lora_weights = [self.load_network_weights(path, "musubi_tuner.networks.lora_minimax_h3") for path in base_weight_paths]
        base_lora_multipliers = list(getattr(args, "base_weights_multiplier", None) or [])
        if len(base_lora_multipliers) > len(base_lora_weights):
            logger.warning(
                f"--base_weights_multiplier lists {len(base_lora_multipliers)} values for {len(base_lora_weights)} "
                "--base_weights; the extra ones are ignored"
            )
        base_lora_multipliers.extend([1.0] * (len(base_lora_weights) - len(base_lora_multipliers)))
        base_lora_multipliers = base_lora_multipliers[: len(base_lora_weights)]
        backend_kwargs = dict(
            model=Path(dit_path),
            device=str(loading_device),
            dtype=model_utils.dtype_to_str(self.dit_dtype),
            mode=args.h3_training_mode,
            attention_mode=attn_mode,
            split_attention=split_attn,
            fp8_scaled=bool(args.fp8_base),
            adaln_rank=args.h3_adaln_rank,
            fp8_quantization_mode=args.h3_fp8_quantization_mode,
            convrot_int8=bool(args.h3_convrot_int8),
            convrot_int8_bwd=args.h3_convrot_int8_bwd,
            convrot_int8_fwd=args.h3_convrot_int8_fwd,
            quantization_device=str(accelerator.device),
            int8_convrot=bool(args.int8_convrot_base),
            target_device=str(accelerator.device),
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0) or 0),
            block_swap_h2d_only=bool(getattr(args, "block_swap_h2d_only", False)),
        )
        reference_image_short_edge = int(getattr(args, "reference_image_short_edge", REFERENCE_IMAGE_SHORT_EDGE))
        if reference_image_short_edge != REFERENCE_IMAGE_SHORT_EDGE:
            backend_kwargs["reference_image_short_edge"] = reference_image_short_edge
        reference_image_size_mode = getattr(args, "reference_image_size_mode", "short_edge")
        reference_image_max_pixels = int(getattr(args, "reference_image_max_pixels", 0) or 0)
        if reference_image_size_mode != "short_edge" or reference_image_max_pixels:
            backend_kwargs["reference_image_size_mode"] = reference_image_size_mode
            backend_kwargs["reference_image_max_pixels"] = reference_image_max_pixels
        reference_video_short_edge = int(getattr(args, "reference_video_short_edge", REFERENCE_VIDEO_SHORT_EDGE))
        reference_video_max_pixels = int(getattr(args, "reference_video_max_pixels", REFERENCE_VIDEO_MAX_PIXELS))
        if reference_video_short_edge != REFERENCE_VIDEO_SHORT_EDGE or reference_video_max_pixels != REFERENCE_VIDEO_MAX_PIXELS:
            backend_kwargs["reference_video_short_edge"] = reference_video_short_edge
            backend_kwargs["reference_video_max_pixels"] = reference_video_max_pixels
        reference_video_fps = float(getattr(args, "reference_video_fps", REFERENCE_VIDEO_FPS) or REFERENCE_VIDEO_FPS)
        if reference_video_fps != REFERENCE_VIDEO_FPS:
            backend_kwargs["reference_video_fps"] = reference_video_fps
        text_visual_max_pixels = int(getattr(args, "h3_text_visual_max_pixels", 0) or 0)
        if text_visual_max_pixels:
            backend_kwargs["text_visual_max_pixels"] = text_visual_max_pixels
        max_caption_tokens = int(getattr(args, "h3_max_caption_tokens", 0) or 0)
        if max_caption_tokens:
            backend_kwargs["max_caption_tokens"] = max_caption_tokens
        if base_lora_weights:
            backend_kwargs["base_lora_weights"] = base_lora_weights
            backend_kwargs["base_lora_multipliers"] = base_lora_multipliers
        self.backend = create_training_backend(**backend_kwargs)
        transformer = self.backend.get_training_transformer()
        if not isinstance(transformer, torch.nn.Module):
            raise TypeError("H3 backend get_training_transformer() must return a torch.nn.Module")
        if args.h3_attn_auto_dispatch:
            transformer.enable_attention_auto_dispatch()
        if base_weight_paths:
            args.base_weights = None
            args.base_weights_multiplier = None
            # Remembered because the argument is cleared here and nothing downstream can
            # tell afterwards that the model it was handed is no longer the checkpoint.
            self._merged_base_weight_paths = list(base_weight_paths)
            accelerator.print("all H3 base weights merged during model loading: " + ", ".join(base_weight_paths))
        return transformer

    @staticmethod
    def _compile_opaque_attention(args) -> bool:
        """Whether compiled blocks call the attention kernel through the Dynamo-opaque wrapper.

        ``inline`` (the default) traces the call as every --compile run did before the option
        existed. ``auto`` hides FlashAttention (its Python binding graph-breaks inside the call,
        which cascades into fragments around every block's kernel) and leaves SDPA traced, since
        it stays in the graph on its own; ``fullgraph`` forbids any break, so auto never hides."""
        mode = getattr(args, "h3_compile_attention", "inline")
        if mode == "opaque":
            return True
        if mode == "inline":
            return False
        return not getattr(args, "sdpa", False) and not getattr(args, "compile_fullgraph", False)

    def compile_transformer(self, args, transformer):
        target_blocks = model_utils.resolve_compile_block_lists(transformer, ("blocks", "token_refiner.blocks"))
        count = sum(len(blocks) for blocks in target_blocks)
        logger.info("MiniMax H3: resolved %d regional torch.compile blocks", count)
        if count == 0:
            raise RuntimeError("--compile set but no H3 transformer blocks were resolved")
        opaque = self._compile_opaque_attention(args)
        set_opaque = getattr(transformer, "set_compile_opaque_attention", None)
        if callable(set_opaque):
            set_opaque(opaque)
            logger.info(
                "MiniMax H3: attention kernel %s to Dynamo (--h3_compile_attention %s)",
                "opaque, one graph break per block at the kernel" if opaque else "inline, traced into the block graph",
                getattr(args, "h3_compile_attention", "inline"),
            )
        return model_utils.compile_transformer(
            args,
            transformer,
            target_blocks,
            disable_linear=self.blocks_to_swap > 0,
        )

    def scale_shift_latents(self, latents):
        # H3 latent caches are written in the model's normalized latent space.
        return latents

    def _base_sigma(
        self,
        args: argparse.Namespace,
        noise_scheduler,
        timesteps: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Recover the *unshifted* schedule coordinate for this step.

        Both branches are unshifted only because ``--discrete_flow_shift`` is
        pinned to 1.0 (enforced in ``handle_model_specific_args``): the direct
        modes never apply it, and the scheduler branch builds
        ``FlowMatchDiscreteScheduler(shift=discrete_flow_shift)``. The chosen
        ``--timestep_sampling`` therefore only picks the *shape* of the base
        distribution; H3's own shifts are applied downstream.

        Both branches resolve in fp32. Reading the schedule in the DiT dtype
        would quantize the base coordinate to the ~256 distinct BF16 values in
        [0, 1] before H3's 12/3 shifts are applied downstream.
        """
        if args.timestep_sampling in _DIRECT_SIGMA_SAMPLING:
            return ((timesteps.to(device=device, dtype=torch.float32) - 1.0) / 1000.0).clamp(0.0, 1.0)
        return get_sigmas(noise_scheduler, timesteps, device, n_dim=1, dtype=torch.float32)

    def _apply_frame_sigma_jitter(self, args, inputs, video_latents, video_noise, base_sigma, is_image):
        """Give each latent frame its own noise level around the shared schedule.

        One sigma per step supervises one point of the schedule per step. Drawing
        a nearby sigma per frame supervises a spread of the schedule in the same
        forward, which is worth most when data is scarce. The flow target
        ``x0 - noise`` does not depend on sigma, so only the noised input and the
        per-row timesteps change.
        """
        if self._frame_sigma_jitter <= 0 or inputs.video is None or is_image:
            return inputs, None
        frames = video_latents.shape[2]
        rows_per_frame_h, rows_per_frame_w = VIDEO_DIT_PATCH_SIZE[-2:]
        rows_per_frame = (video_latents.shape[-2] // rows_per_frame_h) * (video_latents.shape[-1] // rows_per_frame_w)
        base = float(base_sigma.reshape(-1)[0])
        epsilon = min(1e-4, self._frame_sigma_jitter * 0.5)
        lower = max(epsilon, base - self._frame_sigma_jitter)
        upper = min(1.0 - epsilon, base + self._frame_sigma_jitter)
        frame_base = lower + torch.rand(frames, device="cpu") * (upper - lower)
        frame_sigma = shift_sigma(frame_base, 1.0 if is_image else args.h3_shift_video)
        sigma = frame_sigma.to(device=video_latents.device, dtype=video_latents.dtype).view(1, 1, frames, 1, 1)
        noisy = (1.0 - sigma) * video_latents + sigma * video_noise
        row_timestep = (1.0 - frame_sigma).repeat_interleave(rows_per_frame)
        mean_sigma = frame_sigma.mean().reshape(1).to(device=inputs.video_sigma.device, dtype=inputs.video_sigma.dtype)
        return (
            replace(
                inputs,
                video=noisy,
                video_sigma=mean_sigma,
                video_timestep=1.0 - mean_sigma,
                video_frame_sigma=frame_sigma,
            ),
            row_timestep,
        )

    def _draw_spatial_density_scale(self):
        """Draw this step's spatial packing density.

        H3's spatial RoPE is area-normalized, so token spacing is fixed by the
        latent area and a single-resolution dataset teaches exactly one spacing.
        Perturbing the effective area per step synthesizes the range of spacings
        a multi-resolution dataset would supply, without re-caching anything.

        The factor is drawn log-uniformly so denser and sparser packing are
        equally likely, and one draw covers every spatial grid in the sequence so
        reference and target rows stay in coordinate correspondence.
        """
        if self._spatial_density_jitter <= 0:
            return None
        span = math.log1p(self._spatial_density_jitter)
        return float(torch.exp((torch.rand((), device="cpu") * 2 - 1) * span))

    def _resolve_keyframe_anchors(self, video):
        """Resolve this step's conditioning anchors.

        Returns ``(anchors, indices)``. ``anchors`` is what the packer receives
        and keeps ``"first"``/``"last"`` as themselves, because ``"last"`` names
        the final *pixel* frame while the integer ``frames - 1`` names the final
        latent window's start -- collapsing them would silently move the released
        anchor. ``indices`` says which latent frame supplies the content, where
        ``"last"`` does take the final window.
        """
        if video is None and (self._keyframe_anchors or self._keyframe_random_count):
            raise ValueError("H3 keyframe conditioning requires a video target")
        if video is None:
            return (), ()
        frames = video.shape[-3]
        if self._keyframe_random_count:
            count = min(self._keyframe_random_count, frames)
            drawn = sorted(int(index) for index in torch.randperm(frames, device="cpu")[:count])
            return tuple(drawn), tuple(drawn)
        anchors: list[int | str] = []
        indices: list[int] = []
        for anchor in self._keyframe_anchors:
            index = 0 if anchor == "first" else frames - 1 if anchor == "last" else int(anchor)
            if isinstance(anchor, int) and anchor < 0:
                index = frames + anchor
            if not 0 <= index < frames:
                raise ValueError(f"H3 keyframe anchor {anchor} is outside the {frames} target latent frames")
            anchors.append(index if isinstance(anchor, int) and anchor < 0 else anchor)
            indices.append(index)
        # Mirror the packer's identity rule: "first" and an explicit 0 name the
        # same coordinate and collide, while "last" is its own coordinate and
        # never collides with an index.
        identities = [0 if anchor == "first" else anchor for anchor in anchors]
        if len(set(identities)) != len(identities):
            raise ValueError("H3 keyframe anchors resolved to duplicate latent frames")
        order = sorted(range(len(indices)), key=lambda position: indices[position])
        return tuple(anchors[position] for position in order), tuple(indices[position] for position in order)

    def _resolve_guide_specs(self, video, audio):
        """Resolve authored pixel-frame guide recipes against this target."""
        specs = getattr(self, "_guide_specs", ())
        if not specs:
            return ()
        if video is None:
            raise ValueError("H3 guide training requires a video target timeline")
        latent_frames = int(video.shape[-3])
        pixel_spans = tuple((1, 4, 4, 4, 4)[index % 5] for index in range(latent_frames))
        boundaries = [0]
        for span in pixel_spans:
            boundaries.append(boundaries[-1] + span)
        frame_count = boundaries[-1]
        resolved = []
        starts = set()
        for authored_start, video_length, audio_length in specs:
            start = authored_start if authored_start >= 0 else frame_count + authored_start
            if not 0 <= start < frame_count:
                raise ValueError(f"H3 guide frame {authored_start} is outside the {frame_count}-frame target")
            if start in starts:
                raise ValueError(f"H3 guide frame {start} is listed twice")
            starts.add(start)
            video_start = None
            if video_length:
                if start not in boundaries[:-1]:
                    raise ValueError(
                        f"H3 visual guide frame {start} is not a cached VAE-window boundary; "
                        f"use one of {boundaries[:-1]} or cache an external guide clip"
                    )
                video_start = boundaries.index(start)
                if video_start + video_length > latent_frames:
                    raise ValueError(f"H3 {video_length}-latent visual guide at frame {start} does not fit the target")
            audio_start = math.floor((5.0 / 3.0) * start)
            if audio_length:
                if audio is None:
                    raise ValueError("H3 audio guide training requires target audio latents")
                if (5 * start) % 3:
                    raise ValueError(
                        f"H3 audio guide frame {start} is not a cached audio-latent boundary; "
                        "target-derived audio guide starts must be multiples of 3 pixel frames"
                    )
                if audio_start + audio_length > audio.shape[-1]:
                    raise ValueError(f"H3 {audio_length}-latent audio guide at frame {start} does not fit the target")
            resolved.append((MiniMaxH3GuideGeometry(start, video_length, audio_length), video_start, audio_start))
        return tuple(resolved)

    @staticmethod
    def _clean_latents(noisy, target, sigma):
        """Recover x0 from the noised latents and their flow target."""
        shape = [1] * noisy.ndim
        shape[0] = sigma.shape[0]
        return noisy + sigma.to(device=noisy.device, dtype=noisy.dtype).view(shape) * target

    @staticmethod
    def _dataset_observed_mask(batch, inputs):
        """Read the item's authored observed mask out of the batch.

        Dataset masks are static per item, so they are decoded and reduced to the
        latent grid by the collator; here they only have to be checked against the
        latents they claim to mask.
        """
        mask = None if batch is None else batch.get(H3_CONDITIONING_MASK_KEY)
        if mask is None:
            raise ValueError(
                "--h3_mask_mode dataset reads the observed region from the dataset, so every dataset needs "
                "conditioning_mask_directory or a per-record conditioning_mask_path"
            )
        mask = mask.to(device="cpu", dtype=torch.bool)
        if mask.ndim == 4:
            # Observed rows are one flag per packed row, shared by the whole
            # batch, so a batch may only carry one authored region.
            if mask.shape[0] > 1 and not bool((mask == mask[:1]).all()):
                raise ValueError(
                    "--h3_mask_mode dataset pins one observed region per step; a batch whose items carry "
                    "different masks needs batch_size = 1"
                )
            mask = mask[0]
        if mask.ndim != 3:
            raise ValueError(f"H3 conditioning mask must be [frames, height, width], got {tuple(mask.shape)}")
        expected = tuple(int(value) for value in inputs.video.shape[-3:])
        if tuple(mask.shape) != expected:
            raise ValueError(f"H3 conditioning mask has shape {tuple(mask.shape)} for {expected} video latents")
        return mask

    def _draw_step_mask(self, inputs, patch_size, batch=None):
        """Draw one conditioning mask per step, shared by every forward it needs.

        The teacher, empty and trainable branches must all see the same observed
        region; drawing per forward would let them disagree about what is given.
        ``dataset`` mode reads the region the dataset authored instead of drawing
        one, which is the only difference: everything downstream is identical.
        """
        if self._mask_mode == "off" and not self._mask_audio:
            return None
        # A mixed run draws one recipe per step; a step that drew extension or the
        # plain objective must present no masked rows at all.
        if self._step_recipe not in (None, "mask"):
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        video_rows = audio_rows = video_latent = audio_latent = None
        if self._mask_mode != "off" and inputs.video is not None:
            frames, height, width = inputs.video.shape[-3:]
            if self._mask_mode == "dataset":
                # The authored mask marks observed context; the samplers return
                # True where the model generates, so it enters inverted.
                latent = ~self._dataset_observed_mask(batch, inputs)
            else:
                latent = sample_video_mask(
                    mode=self._mask_mode,
                    latent_frames=frames,
                    latent_height=height,
                    latent_width=width,
                    generator=generator,
                    minimum=self._mask_bounds[0],
                    maximum=self._mask_bounds[1],
                )
            rows = video_mask_to_rows(latent, patch_size)
            # A patch counts as generated when any latent inside it is, so the
            # loss must score the whole patch rather than the drawn region.
            video_latent = rows_to_latent_video_mask(
                rows, latent_frames=frames, latent_height=height, latent_width=width, patch_size=patch_size
            )
            video_rows = ~rows
        if self._mask_audio and inputs.audio is not None:
            latent = sample_audio_mask(
                num_audio_latents=inputs.audio.shape[-1],
                generator=generator,
                minimum=self._mask_bounds[0],
                maximum=self._mask_bounds[1],
            )
            audio_rows = ~audio_mask_to_rows(latent, channels=AUDIO_CHANNELS)
            audio_latent = latent
        if video_rows is None and audio_rows is None:
            return None
        return SimpleNamespace(video_rows=video_rows, audio_rows=audio_rows, video_latent=video_latent, audio_latent=audio_latent)

    @staticmethod
    def _mask_to_loss(mask, target, generated, *, axis: int):
        """Restrict a modality's loss to the generated region."""
        if generated is None or target is None:
            return mask
        shape = [1] * target.ndim
        if generated.ndim == 1:
            shape[axis] = generated.shape[0]
        else:
            shape[-generated.ndim :] = list(generated.shape)
        keep = generated.to(device=target.device).view(shape).expand_as(target)
        if mask is None:
            return keep
        mask = mask.to(device=target.device)
        return torch.where(keep, mask, torch.zeros((), device=mask.device, dtype=mask.dtype))

    @staticmethod
    def _clean_context(noisy, target, sigma, context_length: int, *, axis: int):
        """Recover the clean leading latents the observed context must present.

        The packed rows are already noised, so the context has to be rebuilt.
        H3's flow gives it exactly: ``x_t = (1 - s) * x0 + s * noise`` and
        ``target = x0 - noise`` imply ``x0 = x_t + s * target``, so no separate
        cache of the clean span is needed.
        """
        if noisy is None or target is None:
            raise ValueError("H3 extension needs both the noisy latents and their flow target")
        shape = [1] * noisy.ndim
        shape[0] = sigma.shape[0]
        clean = noisy + sigma.to(device=noisy.device, dtype=noisy.dtype).view(shape) * target
        index = [slice(None)] * noisy.ndim
        index[axis] = slice(0, context_length)
        return clean[tuple(index)].contiguous()

    @staticmethod
    def _extension_masked(mask, target, context_length: int, *, axis: int):
        """Drop the observed leading context from a modality's loss mask."""
        if not context_length or target is None:
            return mask
        length = target.shape[axis]
        if context_length >= length:
            raise ValueError(f"H3 extension context {context_length} covers the whole {length}-long target")
        keep = torch.ones(length, dtype=torch.bool, device=target.device)
        keep[:context_length] = False
        shape = [1] * target.ndim
        shape[axis] = length
        keep = keep.view(shape).expand_as(target)
        if mask is None:
            return keep
        mask = mask.to(device=target.device)
        return torch.where(keep, mask, torch.zeros((), device=mask.device, dtype=mask.dtype))

    def _sample_weight(self, args: argparse.Namespace, sigma: torch.Tensor, modality: str = "video") -> torch.Tensor | None:
        """Per-sample loss weight at this modality's SHIFTED sigma; ``None`` means uniform."""
        weight = None
        if args.weighting_scheme == "sigma_sqrt":
            # H3 samples a continuous base coordinate, so the generic
            # sigma^-2 weighting has no finite upper bound. Clamp sigma at
            # the equivalent configured maximum before taking the inverse.
            sigma_floor = float(args.h3_sigma_sqrt_max_weight) ** -0.5
            weight = sigma.clamp_min(sigma_floor).pow(-2.0)
        elif args.weighting_scheme == "cosmap":
            weight = 2.0 / (math.pi * (1.0 - 2.0 * sigma + 2.0 * sigma.square()))
        curve = getattr(self, "_measured_variance_curve", None)
        if curve and modality in curve:
            sigmas, weights = curve[modality]
            measured = interpolate_curve(sigmas, weights, sigma)
            weight = measured if weight is None else weight * measured
        return weight

    def _validate_rollout_args(self, args: argparse.Namespace) -> None:
        """Reject every incoherent rollout-supervision configuration before loading.

        Rollout supervision replaces the *video* half of the data objective on an
        active step with a regression onto a privileged teacher evaluated at a
        state the model's own sampler produced from noise. Two consequences drive
        everything below. First, on an active step there is no data state for
        video at all, so every recipe that presents part of the target as observed
        has nothing to observe. Second, the teacher forward needs the trainable
        network switched off, which only an adapter can do.
        """
        rollout = bool(getattr(args, "h3_rollout_supervision", False))
        teacher_config = getattr(args, "h3_rollout_teacher_config", None)
        # KNOWN WEAKNESS, left as it was found: "did the user write this flag" is a
        # comparison against the default value, so a dial written with exactly its
        # default slips through the check below without --h3_rollout_supervision.
        # A sentinel default would settle it for the command line, but these
        # namespaces are also built programmatically with concrete values, and a
        # presence test then rejects a caller that set nothing unusual.
        flags = {
            "--h3_rollout_teacher_config": teacher_config is not None,
            "--h3_rollout_probability": args.h3_rollout_probability != _ROLLOUT_PROBABILITY_DEFAULT,
            "--h3_rollout_steps": args.h3_rollout_steps != _ROLLOUT_STEPS_DEFAULT,
            "--h3_rollout_window": args.h3_rollout_window != _ROLLOUT_WINDOW_DEFAULT,
            "--h3_rollout_stop_shifted": bool(getattr(args, "h3_rollout_stop_shifted", False)),
            "--h3_rollout_teacher_privilege": getattr(args, "h3_rollout_teacher_privilege", "auto") != "auto",
            "--h3_rollout_fused_teacher": bool(getattr(args, "h3_rollout_fused_teacher", False)),
            "--h3_rollout_field_floor": float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0) != 0.0,
            "--h3_rollout_field_floor_direction": getattr(args, "h3_rollout_field_floor_direction", "self") != "self",
            "--h3_rollout_field_floor_sigma_max": float(getattr(args, "h3_rollout_field_floor_sigma_max", 1.0)) != 1.0,
            "--h3_rollout_stop_min": float(getattr(args, "h3_rollout_stop_min", 0.0) or 0.0) != 0.0,
            "--h3_rollout_prefix": getattr(args, "h3_rollout_prefix", "student") != "student",
            "--h3_rollout_null_anchor_weight": float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0) != 0.0,
            "--h3_rollout_field_cap": float(getattr(args, "h3_rollout_field_cap", 0.0) or 0.0) != 0.0,
        }
        if not rollout:
            for flag, changed in flags.items():
                if changed:
                    raise ValueError(f"{flag} requires --h3_rollout_supervision")
            return

        if teacher_config is None:
            raise ValueError(
                "--h3_rollout_supervision requires --h3_rollout_teacher_config: the objective is defined by a "
                "teacher conditioned on the target's own frames, and there is no default for that cache"
            )
        if not Path(teacher_config).is_file():
            raise FileNotFoundError(f"--h3_rollout_teacher_config file not found: {teacher_config}")
        if args.h3_rollout_steps < 1:
            raise ValueError("--h3_rollout_steps must be at least 1")
        if not 1 <= args.h3_rollout_window <= MAX_ROLLOUT_WINDOW:
            raise ValueError(f"--h3_rollout_window must lie in [1, {MAX_ROLLOUT_WINDOW}]")
        if not math.isfinite(args.h3_rollout_probability) or not 0 < args.h3_rollout_probability <= 1:
            raise ValueError("--h3_rollout_probability must be finite and lie in (0, 1]")
        if args.h3_video_loss_weight <= 0:
            raise ValueError("--h3_rollout_supervision trains the video field and therefore needs --h3_video_loss_weight > 0")
        field_floor = float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0)
        if not math.isfinite(field_floor) or field_floor < 0:
            # Gated on "> 0" below, so a negative weight would configure the floor
            # and then quietly train without it.
            raise ValueError("--h3_rollout_field_floor must be finite and non-negative; 0 disables it")
        stop_min = float(getattr(args, "h3_rollout_stop_min", 0.0) or 0.0)
        if not math.isfinite(stop_min) or not 0 <= stop_min < 1:
            raise ValueError("--h3_rollout_stop_min must be finite and lie in [0, 1)")
        floor_sigma_max = float(getattr(args, "h3_rollout_field_floor_sigma_max", 1.0))
        if not math.isfinite(floor_sigma_max) or not 0 < floor_sigma_max <= 1:
            raise ValueError("--h3_rollout_field_floor_sigma_max must be finite and lie in (0, 1]")
        rollout_anchor = float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0)
        if not math.isfinite(rollout_anchor) or rollout_anchor < 0:
            raise ValueError("--h3_rollout_null_anchor_weight must be finite and non-negative; 0 disables it")
        if rollout_anchor > 0 and getattr(args, "h3_adapter_prompt_only", False):
            raise ValueError(
                "--h3_rollout_null_anchor_weight holds the student's empty branch, which --h3_adapter_prompt_only "
                "already pins to the frozen one; drop one of the two"
            )
        field_cap = float(getattr(args, "h3_rollout_field_cap", 0.0) or 0.0)
        if field_cap != 0.0 and (not math.isfinite(field_cap) or field_cap <= 1.0):
            raise ValueError("--h3_rollout_field_cap must be finite and above 1 (0 disables it)")
        if field_floor <= 0 and (
            getattr(args, "h3_rollout_field_floor_direction", "self") != "self" or floor_sigma_max != 1.0 or field_cap != 0.0
        ):
            raise ValueError(
                "--h3_rollout_field_floor_direction, --h3_rollout_field_floor_sigma_max and --h3_rollout_field_cap shape "
                "the field floor and need --h3_rollout_field_floor above 0"
            )
        if (
            field_floor > 0
            and float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0) <= 0
            and float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0) <= 0
        ):
            # The floor measures the student's prompted prediction against the
            # FROZEN empty branch. What inference amplifies is the gap to the
            # student's OWN empty branch, and nothing in the floor stops that
            # branch from drifting toward the prompted one; the anchor is what
            # holds it. Without the anchor the floor is a displacement floor on
            # the prompted branch alone, which is a weaker statement.
            logger.warning(
                "--h3_rollout_field_floor holds the prompted prediction's distance from the FROZEN empty branch; the "
                "field inference amplifies is the distance to the student's own empty branch, which only an anchor "
                "(--h3_guidance_null_anchor_weight or --h3_rollout_null_anchor_weight) holds. Without one the floor "
                "does not bound that field"
            )
        if getattr(args, "h3_rollout_fused_teacher", False) and getattr(args, "h3_int8_attention", "off") == "aux":
            # Fusing puts the student and the teacher inside one pass over the
            # blocks, where the attention kernel is chosen per block and not per
            # forward. "aux" asks for INT8 on the auxiliary forwards ONLY, which
            # that pass cannot express; "train" and "off" apply to both arms alike
            # and compose with fusion unchanged.
            raise ValueError(
                "--h3_rollout_fused_teacher shares one pass over the blocks with the student, so --h3_int8_attention aux "
                "cannot select INT8 for the teacher arm alone; use --h3_int8_attention train or off, or drop the fusion"
            )
        if getattr(args, "h3_rollout_fused_teacher", False) and (
            float(getattr(args, "h3_block_sparse_kv_fraction", 0.0) or 0.0) > 0
            or float(getattr(args, "h3_block_sparse_threshold", 0.0) or 0.0) > 0
        ):
            # The block-sparse tile plan is built per packed sequence and stored on
            # the attention modules themselves. The fused pass prepares every arm
            # before it runs the first block, so the plan installed last -- the
            # teacher's, whose privileged sequence is usually the longer one -- is
            # the one every arm's attention would read.
            raise ValueError(
                "--h3_rollout_fused_teacher runs the student and the teacher through one pass over the blocks, whose "
                "block-sparse attention plan is built per sequence and shared by the modules; with "
                "--h3_block_sparse_kv_fraction or --h3_block_sparse_threshold above 0 every arm would attend under "
                "the plan of the arm prepared last. Drop the fusion or the block-sparse attention"
            )
        # The guidance pair is allowed. Round 1 measured why: the rollout preserved
        # the null field exactly where it supervised -- the high shifted sigmas its
        # stop draw reaches -- and eroded the mid band (gap ratio 0.14 at sigma .6)
        # because an inactive step ran plain flow matching and nothing held the
        # field there. The hybrid gives those steps the guidance objective back.
        # The two never claim one slot: on an ACTIVE step the video target is the
        # teacher's and the guidance correction applies to audio only, which is the
        # half the rollout leaves on its data loss anyway.
        if args.h3_observed_modality is not None:
            raise ValueError(
                "--h3_rollout_supervision generates both modalities from noise, so it has no observed side; "
                "drop --h3_observed_modality"
            )
        conditioning_flags = {
            "--h3_mask_mode/--h3_mask_audio": args.h3_mask_mode != "off" or args.h3_mask_audio,
            "--h3_extension_video_frames/--h3_extension_audio_latents": bool(
                args.h3_extension_video_frames or args.h3_extension_audio_latents
            ),
            "--h3_keyframe_anchors/--h3_keyframe_random_count": bool(args.h3_keyframe_anchors or args.h3_keyframe_random_count),
            "--h3_guide_specs": bool(getattr(args, "h3_guide_specs", "")),
        }
        for flag, configured in conditioning_flags.items():
            if configured:
                raise ValueError(
                    f"{flag} presents part of the target as observed, which a rollout from pure noise has no clean "
                    "latents to build; disable it or drop --h3_rollout_supervision"
                )
        if args.h3_frame_sigma_jitter > 0:
            raise ValueError("--h3_frame_sigma_jitter gives each frame its own noise level, which a rollout state has no room for")
        if args.crepa is not None:
            # CREPA captures activations of one trainable forward per step and
            # holds them until backward. The window's forwards would overwrite the
            # capture the data forward made, silently aligning against the wrong
            # state.
            raise ValueError("--crepa captures one trainable forward per step and cannot be combined with --h3_rollout_supervision")

    @staticmethod
    def _runtime_network_toggle(accelerator, network, requirement: str):
        """Return the ``set_enabled`` of the trainable network for a frozen-base forward."""
        if network is None:
            raise ValueError(f"{requirement} requires a trainable network")
        unwrapped_network = accelerator.unwrap_model(network)
        set_enabled = getattr(unwrapped_network, "set_enabled", None)
        if not callable(set_enabled):
            raise TypeError(f"{requirement} requires a network with set_enabled()")

        # Every frozen-base forward in this trainer takes its toggle from here, so
        # recording the requested state on the network is enough for the
        # prompt-only wrapper in ``_predict`` to know whether the adapter is
        # already off and leave it alone.
        def tracked(enabled: bool) -> None:
            unwrapped_network._h3_adapter_enabled = bool(enabled)
            set_enabled(enabled)

        return tracked

    @staticmethod
    @contextmanager
    def _trainable_block_swap(transformer, active: bool):
        """Return to the TRAINING swap contract for one graph-carrying forward.

        The inverse of ``_auxiliary_block_swap``. The auxiliary section brackets a
        run of no-grad forwards into the cyclic forward-only schedule, which is
        correct for every forward that has no backward -- but the null anchor's
        student pass lives inside that same section and DOES carry a graph. Under
        classic block swap its activations would then be produced under the
        forward-only contract while backward expects the training one, which
        restores the wrong blocks. Bracketing just that forward back costs two
        mode switches and keeps the surrounding no-grad run unchanged.
        """
        if active:
            transformer.switch_block_swap_for_training()
        try:
            yield
        finally:
            if active:
                transformer.switch_block_swap_for_inference()

    @staticmethod
    @contextmanager
    def _auxiliary_block_swap(transformer, active: bool):
        """Run one no-grad forward on the cyclic forward-only swap schedule.

        Classic block swap in training mode leaves the forward prefix on CPU for
        its backward hooks to restore, so an auxiliary forward with no backward
        must enter forward-only mode and return to the training layout before the
        next graph-carrying forward. The rollout interleaves the two, so unlike
        the guidance branch it brackets each auxiliary forward rather than one
        contiguous group.
        """
        if active:
            transformer.switch_block_swap_for_inference()
        try:
            yield
        finally:
            if active:
                transformer.switch_block_swap_for_training()

    def _rollout_state(
        self,
        *,
        video: torch.Tensor | None,
        audio: torch.Tensor | None,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> H3JointNoisyInputs:
        """One point of a rollout, in the shape ``_predict`` reads.

        The flow targets are ``None`` on purpose: a rollout state has no data
        behind it, so there is no ``x0 - noise`` to point at. Every consumer of
        those fields -- keyframe anchors, guides, masked conditioning, extension
        context -- is rejected at argument time for exactly this reason.
        """
        return H3JointNoisyInputs(
            video=video,
            audio=audio,
            video_target=None,
            audio_target=None,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
            video_timestep=1.0 - video_sigma,
            audio_timestep=1.0 - audio_sigma,
        )

    @staticmethod
    def _frozen_arm_context(set_enabled):
        """The context a frozen arm of a fused forward is evaluated under.

        Re-entered around every stage of that arm -- its embedding, each of its
        block calls, its final layer -- so it must be cheap and idempotent.
        ``torch.no_grad()`` is what keeps the teacher's activations out of the
        graph, and disabling the adapter is what makes it the frozen base rather
        than a second copy of the student.
        """

        @contextmanager
        def enter():
            with torch.no_grad():
                set_enabled(False)
                try:
                    yield
                finally:
                    set_enabled(True)

        return enter

    def _fused_rollout_pair(
        self,
        accelerator: Accelerator,
        transformer,
        batch: dict,
        teacher_presentation: dict,
        state,
        *,
        conditioning: str,
        set_enabled,
        fork_devices: list,
        frozen_base: bool = False,
        frozen_empty: bool = False,
        student_empty: bool = False,
    ) -> list[H3ModelPrediction]:
        """One supervised sub-step evaluated in a single pass over the blocks.

        Returns ``[student, teacher]``, followed by the frozen base's prompted and
        empty predictions at the same state when ``frozen_base`` is set: the two
        arms ``--h3_rollout_field_floor`` reads, riding the same block loop.

        The student and the teacher pack different sequences -- the privileged
        presentation is the longer one, which is the entire point of it -- so they
        cannot share a batch axis. They share the *block loop* instead: each
        swapped block is streamed from CPU once and both arms consume it before it
        is released, which is where the time goes under ``--blocks_to_swap``.

        The two arms never touch each other's tensors, so attention is isolated by
        construction rather than by a mask, and nothing about either arm's own
        attention changes. The teacher arm's rows stay out of the graph because
        its whole evaluation -- embedding, blocks, final layer -- runs inside
        ``torch.no_grad()`` with the adapter off.

        No auxiliary block-swap bracket is taken here on purpose: the fused call
        runs on the *training* layout that the student's graph-carrying arm needs,
        and the teacher simply rides it. That is the bracket the unfused path pays
        twice per sub-step and this one pays not at all.
        """
        run = self._frozen_arm_context(set_enabled)
        student = H3FusedArm(call=self._predict_call(accelerator, batch, state, conditioning=conditioning))
        teacher = H3FusedArm(
            call=self._predict_call(
                accelerator,
                teacher_presentation,
                state,
                # The teacher's privilege lives in its prompt presentation; there
                # is no privileged null branch, so an active step is always a
                # prompted one.
                conditioning="prompt",
            ),
            # Forking the RNG leaves the teacher's stochastic conditioning
            # identical to the student's at this state, so the pair describes one
            # state and not two. It brackets the teacher's packed-sequence build,
            # which is the only stage that draws; the block loop draws nothing,
            # which is why ``run`` must not fork and does not.
            build=lambda: _forked_frozen_build(fork_devices, set_enabled),
            run=run,
        )
        arms = [student, teacher]
        # Frozen arms after the pair: the base's prompted and empty predictions
        # for the floor, or the empty one alone for the rollout anchor.
        frozen_branches = ("prompt", "empty") if frozen_base else (("empty",) if frozen_empty else ())
        arms.extend(
            H3FusedArm(
                call=self._predict_call(accelerator, batch, state, conditioning=branch),
                build=lambda: _forked_frozen_build(fork_devices, set_enabled),
                run=run,
            )
            for branch in frozen_branches
        )
        if student_empty:
            # Graded, adapter on: the student's own empty branch at this state.
            arms.append(H3FusedArm(call=self._predict_call(accelerator, batch, state, conditioning="empty")))
        return self._predict_fused(accelerator, transformer, arms)

    def _teacher_presentation(self, batch: dict) -> dict:
        """The privileged teacher's conditioning, presented through this batch.

        The reference bundle is the teacher's under variant B and empty under
        variant A; either way the target latents stay the student's, so the two
        arms are always compared at one and the same state -- which is what makes
        the supervision a statement about the model rather than about two
        different clips.
        """
        if self._rollout_teacher is None:
            raise RuntimeError("--h3_rollout_supervision was configured without a teacher cache")
        item_key = batch_item_key(batch)
        return teacher_batch(
            batch,
            self._rollout_teacher.entries(item_key),
            self._rollout_teacher.reference_entries(item_key),
        )

    def _rollout_supervision(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict,
        *,
        video_latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        video_shift: float,
        audio_shift: float,
        conditioning: str,
        auxiliary_block_swap: bool,
    ) -> tuple[
        list[tuple[H3ModelPrediction, H3ModelPrediction]],
        float,
        list[torch.Tensor],
        list[tuple[H3ModelPrediction | None, H3ModelPrediction]] | None,
        list[H3ModelPrediction] | None,
    ]:
        """Roll the student's own sampler out from noise and supervise the tail.

        Returns one ``(student, teacher)`` prediction pair per supervision
        sub-step, the unshifted base sigma the rollout stopped at, each
        sub-step's shifted video sigma so the loss can weight on the noise level
        it was actually taken at rather than the data step's, and -- under
        ``--h3_rollout_field_floor`` or ``--h3_rollout_null_anchor_weight`` -- one
        ``(base_prompted, base_empty)`` pair of frozen predictions per sub-step at
        that same state (``base_prompted`` is ``None`` when only the anchor asked),
        else ``None``; and under the rollout anchor the student's own graded
        empty-prompt prediction per sub-step, else ``None``.
        Both branches of every pair are evaluated at the SAME state:
        the student with the ordinary conditioning it trains under, the teacher --
        the frozen base, adapter disabled -- with the privileged presentation that
        also shows the conditioner frames of the target clip.

        Cost, per active step: ``steps`` no-grad forwards to reach the supervised
        state -- walked by the student (on-policy, default) or by the frozen
        privileged teacher (``--h3_rollout_prefix teacher``) -- then ``window``
        grad forwards and ``window`` no-grad teacher forwards. Gradient checkpointing applies to the grad forwards without
        anything extra: the transformer gates its checkpoint wrapper on
        ``torch.is_grad_enabled()``, which is exactly what separates the two
        populations here.

        ``--h3_rollout_fused_teacher`` folds each sub-step's two forwards into one
        pass over the blocks. It changes no arithmetic -- both arms compute what
        they computed before, bit for bit -- only the order the weights are
        touched in, which is what ``--blocks_to_swap`` charges for.
        """
        teacher_presentation = self._teacher_presentation(batch)
        set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_rollout_supervision")
        int8_context = getattr(transformer, "int8_attention_context", None)
        fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        field_floor = float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0) > 0
        rollout_anchor = float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0) > 0
        if field_floor or rollout_anchor:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise ValueError(
                    "--h3_rollout_field_floor evaluates the frozen base's EMPTY branch at every supervised state and "
                    "needs the empty presentation in the student's text cache; re-cache with --cache_guidance_empty "
                    f"(missing {', '.join(missing_empty)})"
                )

        stop_sigma = self._draw_rollout_stop_sigma(args, video_shift)
        base = torch.tensor(rollout_base_sigmas(stop_sigma, args.h3_rollout_steps, args.h3_rollout_window), dtype=torch.float32)
        # Each modality rides its own shift off the shared unshifted coordinate,
        # exactly as a data step does, so the joint rollout stays synchronized.
        video_sigmas = shift_sigma(base, video_shift)
        audio_sigmas = shift_sigma(base, audio_shift)

        video_state = None if video_latents is None else self._rollout_noise(video_latents)
        audio_state = None if audio_latents is None else self._rollout_noise(audio_latents)

        def state_at(index: int) -> H3JointNoisyInputs:
            return self._rollout_state(
                video=video_state,
                audio=audio_state,
                video_sigma=video_sigmas[index : index + 1],
                audio_sigma=audio_sigmas[index : index + 1],
            )

        def advance(index: int, velocity: H3ModelPrediction) -> None:
            nonlocal video_state, audio_state
            if index + 1 >= base.shape[0]:
                return
            if video_state is not None:
                video_state = euler_advance(video_state, velocity.video, float(video_sigmas[index]), float(video_sigmas[index + 1]))
            if audio_state is not None:
                audio_state = euler_advance(audio_state, velocity.audio, float(audio_sigmas[index]), float(audio_sigmas[index + 1]))

        # Reaching the supervised state costs nothing but forwards: the prefix
        # policy is the current model, adapter included (on-policy), or the frozen
        # privileged teacher under --h3_rollout_prefix teacher, and none of it is
        # differentiated through. The whole prefix shares one swap bracket because
        # it is contiguous; the window below cannot, since its trainable forwards
        # must run on the training layout.
        teacher_prefix = getattr(args, "h3_rollout_prefix", "student") == "teacher"
        with (
            self._auxiliary_block_swap(transformer, auxiliary_block_swap),
            torch.no_grad(),
            int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
        ):
            if teacher_prefix:
                # The frozen privileged teacher walks to the anchor states; the
                # student's window below refines them. Its stochastic conditioning
                # draws are forked away from the student's stream, as for every
                # other frozen forward.
                with torch.random.fork_rng(devices=fork_devices):
                    try:
                        set_enabled(False)
                        for index in range(args.h3_rollout_steps):
                            advance(
                                index,
                                self._predict(
                                    accelerator,
                                    transformer,
                                    teacher_presentation,
                                    state_at(index),
                                    conditioning="prompt",
                                    role="rollout",
                                ),
                            )
                    finally:
                        set_enabled(True)
            else:
                for index in range(args.h3_rollout_steps):
                    advance(
                        index,
                        self._predict(accelerator, transformer, batch, state_at(index), conditioning=conditioning, role="rollout"),
                    )

        fused = bool(getattr(args, "h3_rollout_fused_teacher", False))
        pairs: list[tuple[H3ModelPrediction, H3ModelPrediction]] = []
        bases: list[tuple[H3ModelPrediction | None, H3ModelPrediction]] = []
        student_empties: list[H3ModelPrediction] = []
        for index in range(args.h3_rollout_steps, base.shape[0]):
            state = state_at(index)
            if fused:
                student, teacher, *frozen = self._fused_rollout_pair(
                    accelerator,
                    transformer,
                    batch,
                    teacher_presentation,
                    state,
                    conditioning=conditioning,
                    set_enabled=set_enabled,
                    fork_devices=fork_devices,
                    frozen_base=field_floor,
                    frozen_empty=rollout_anchor and not field_floor,
                    student_empty=rollout_anchor,
                )
                if rollout_anchor:
                    student_empties.append(frozen.pop())
                if field_floor:
                    bases.append((frozen[0], frozen[1]))
                elif rollout_anchor:
                    bases.append((None, frozen[0]))
            else:
                # Captured BEFORE the student runs, then replayed for the teacher.
                # ``fork_rng`` alone would only restore the state the student left
                # behind, so the teacher would draw the CONTINUATION of the
                # student's sequence rather than the same one, and any stochastic
                # conditioning would differ between the two halves of a pair that
                # is supposed to describe a single state.
                entry_cpu_rng = torch.get_rng_state()
                entry_cuda_rng = [torch.cuda.get_rng_state(device) for device in fork_devices]
                student = self._predict(accelerator, transformer, batch, state, conditioning=conditioning)
                if rollout_anchor:
                    # The student's own empty branch at the same state, graded and
                    # with the adapter on; its conditioning draws replay the entry
                    # RNG so the arm describes this state and not the next.
                    with torch.random.fork_rng(devices=fork_devices):
                        torch.set_rng_state(entry_cpu_rng)
                        for device, rng_state in zip(fork_devices, entry_cuda_rng):
                            torch.cuda.set_rng_state(rng_state, device)
                        student_empties.append(self._predict(accelerator, transformer, batch, state, conditioning="empty"))
                with (
                    torch.random.fork_rng(devices=fork_devices),
                    self._auxiliary_block_swap(transformer, auxiliary_block_swap),
                    torch.no_grad(),
                    int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
                ):
                    torch.set_rng_state(entry_cpu_rng)
                    for device, rng_state in zip(fork_devices, entry_cuda_rng):
                        torch.cuda.set_rng_state(rng_state, device)
                    # Disabled inside the try, so a failure in the toggle itself
                    # still reaches the restore rather than leaving the adapter in
                    # whatever state it stopped in.
                    try:
                        set_enabled(False)
                        teacher = self._predict(
                            accelerator,
                            transformer,
                            teacher_presentation,
                            state,
                            # The teacher's privilege lives in its prompt
                            # presentation; there is no privileged null branch, so
                            # an active step is always a prompted one (enforced by
                            # the caller, which skips a caption-dropout step).
                            conditioning="prompt",
                        )
                        if field_floor or rollout_anchor:
                            # The frozen arms describe the same state as the pair
                            # above, so each replays the entry RNG the way the
                            # teacher did rather than continuing its draws.
                            frozen = {}
                            for branch in ("prompt", "empty") if field_floor else ("empty",):
                                torch.set_rng_state(entry_cpu_rng)
                                for device, rng_state in zip(fork_devices, entry_cuda_rng):
                                    torch.cuda.set_rng_state(rng_state, device)
                                frozen[branch] = self._predict(accelerator, transformer, batch, state, conditioning=branch)
                            bases.append((frozen.get("prompt"), frozen["empty"]))
                    finally:
                        set_enabled(True)
            pairs.append((student, teacher))
            # Stop-grad on the advance: the window supervises m independent states,
            # not a differentiable m-step unroll, whose graph would grow with m and
            # whose gradient would flow through states the teacher never scored.
            advance(
                index,
                H3ModelPrediction(
                    video=None if student.video is None else student.video.detach(),
                    audio=None if student.audio is None else student.audio.detach(),
                ),
            )
        window = slice(args.h3_rollout_steps, base.shape[0])
        return (
            pairs,
            stop_sigma,
            [video_sigmas[window][index : index + 1] for index in range(len(pairs))],
            bases if (field_floor or rollout_anchor) else None,
            student_empties if rollout_anchor else None,
        )

    @staticmethod
    def _field_length_ratio(adapted: torch.Tensor, base: torch.Tensor, mask) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample ``||adapted|| / ||base||`` over the authored elements, graph kept.

        Returns the ratio and a per-sample validity flag: a sample with no
        authored element has no field to measure and must not be scored.
        ``vector_norm`` rather than ``sqrt(mean(x^2))`` because the floor bites
        hardest at an exactly zero field, where the latter's derivative is
        singular and the former's is defined (zero).
        """

        def length(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = tensor.float().flatten(1)
            if mask is None:
                count = torch.full((flat.shape[0],), float(flat.shape[1]), device=flat.device)
                return torch.linalg.vector_norm(flat, dim=1) / count.sqrt(), count > 0
            valid = mask.to(device=tensor.device, dtype=torch.float32).expand_as(tensor).flatten(1)
            count = valid.sum(dim=1)
            return torch.linalg.vector_norm(flat * valid, dim=1) / count.clamp_min(1.0).sqrt(), count > 0

        adapted_length, adapted_valid = length(adapted)
        base_length, base_valid = length(base.detach())
        return adapted_length / base_length.clamp_min(1e-6), adapted_valid & base_valid

    @staticmethod
    def _field_projection_ratio(
        adapted: torch.Tensor, base: torch.Tensor, direction: torch.Tensor, mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample ``<adapted, d> / (||d|| ||base||)`` over the authored elements, graph kept.

        The length of the student's field measured along ``direction`` (the
        teacher's field at the same state) against the base's plain length, so a
        field pointing the wrong way scores short however long it is, and the
        gradient on ``adapted`` is the unit teacher direction rather than the
        student's own. Invalid where a sample has no authored element or either
        reference vector vanishes.
        """
        flat_adapted = adapted.float().flatten(1)
        flat_base = base.detach().float().flatten(1)
        flat_direction = direction.detach().float().flatten(1)
        if mask is None:
            valid_elements = torch.ones_like(flat_base)
        else:
            valid_elements = mask.to(device=base.device, dtype=torch.float32).expand_as(base).flatten(1)
        base_norm = torch.linalg.vector_norm(flat_base * valid_elements, dim=1)
        direction_norm = torch.linalg.vector_norm(flat_direction * valid_elements, dim=1)
        projection = (flat_adapted * flat_direction * valid_elements).sum(dim=1) / direction_norm.clamp_min(1e-6)
        valid = (valid_elements.sum(dim=1) > 0) & (base_norm > 0) & (direction_norm > 0)
        return projection / base_norm.clamp_min(1e-6), valid

    def _configured_guidance_scale(self, args, accelerator, inputs):
        """Resolve the guidance scale this step distills towards.

        Without ``--h3_guidance_scale_range`` this is the single authoritative
        ``--h3_guidance_distillation_scale`` float, unchanged. With the range it
        is a per-sample vector drawn uniformly in ``[lower, upper]``, so one step
        teaches the adapter several points of the guidance family at once.

        ``accelerator is None`` marks a replayed evaluation forward: validation
        must stay comparable between runs, so it reads the range's midpoint
        instead of consuming a draw.
        """
        if self._guidance_scale_range is None:
            return args.h3_guidance_distillation_scale
        lower, upper = self._guidance_scale_range
        if accelerator is None:
            return 0.5 * (lower + upper)
        return self._draw_guidance_scale(accelerator, int(inputs.video_sigma.shape[0]))

    def _guidance_loss_inputs(self, args, prediction, empty_prediction, inputs, accelerator=None):
        configured_scale = self._configured_guidance_scale(args, accelerator, inputs)
        per_sample = isinstance(configured_scale, torch.Tensor)
        video_sigma = inputs.video_frame_sigma if inputs.video_frame_sigma is not None else inputs.video_sigma
        if inputs.video_frame_sigma is not None and per_sample:
            # A per-frame sigma indexes frames, a drawn scale indexes samples:
            # the schedule is evaluated on their outer product so the resulting
            # [batch, 1, frames, 1, 1] grid broadcasts over the video latents.
            video_scale = guidance_scale_for_sigma(
                configured_scale.reshape(-1, 1),
                video_sigma.reshape(1, -1),
                args.h3_guidance_loss_schedule,
                sigma_max=float(getattr(args, "h3_guidance_scale_sigma_max", 1.0)),
            ).reshape(configured_scale.shape[0], 1, video_sigma.shape[0], 1, 1)
        else:
            video_scale = guidance_scale_for_sigma(
                configured_scale,
                video_sigma,
                args.h3_guidance_loss_schedule,
                sigma_max=float(getattr(args, "h3_guidance_scale_sigma_max", 1.0)),
            )
            if inputs.video_frame_sigma is not None:
                video_scale = video_scale.reshape(1, 1, -1, 1, 1)
        # The audio field of the released checkpoints is not a clean amplification
        # (its noise-averaged residual is nearly all systematic at the top of the
        # schedule), so the audio half may take its own scale; 1 leaves the audio
        # target plain while the video target keeps the guidance form.
        audio_configured = getattr(args, "h3_guidance_audio_scale", None)
        audio_scale = guidance_scale_for_sigma(
            configured_scale if audio_configured is None else float(audio_configured),
            inputs.audio_sigma,
            args.h3_guidance_loss_schedule,
            sigma_max=float(getattr(args, "h3_guidance_scale_sigma_max", 1.0)),
        )
        if args.h3_guidance_loss_form == "contrastive":
            true_target = H3ModelPrediction(inputs.video_target, inputs.audio_target)
            if args.h3_guidance_cfg_zero:
                # The contrastive form extrapolates away from the null field toward
                # the true flow target, so that target is the reference the null
                # branch is projected onto.
                empty_prediction = cfg_zero_rescaled_empty(empty_prediction, true_target)
            target = contrastive_guidance_target(
                true_target,
                empty_prediction,
                video_scale,
                audio_guidance_scale=audio_scale,
            )
            return prediction, replace(inputs, video_target=target.video, audio_target=target.audio)
        if args.h3_guidance_cfg_zero:
            # The normalized form de-guides the model's own guided field, so the
            # guided prediction is the reference. It enters the projection
            # detached, matching the anchor treatment of the null branch itself.
            empty_prediction = cfg_zero_rescaled_empty(empty_prediction, prediction)
        return (
            guidance_consistent_prediction(
                prediction,
                empty_prediction,
                video_scale,
                audio_guidance_scale=audio_scale,
            ),
            inputs,
        )

    def _predict_call(
        self,
        accelerator: Accelerator,
        batch,
        inputs,
        *,
        conditioning: str | tuple[str, ...],
    ) -> dict:
        """The arguments one training forward is made of, without running it.

        Split out so a fused pair can be described before either arm is
        evaluated: the two arms of a rollout sub-step differ only in the batch
        they present and in their conditioning branch, and everything the step's
        recipe pinned -- the mask, the keyframes, the guides, the extension
        context, the reference modality, the control dropout -- is shared between
        them by construction because it is read from the same step state.
        """
        # Checkpointing is not a per-call argument here: the model reads its own
        # ``gradient_checkpointing`` flag (set once from ``--gradient_checkpointing``)
        # and both ``MiniMaxH3Transformer.forward`` and ``MiniMaxH3TokenRefiner.forward``
        # additionally gate the wrapper on ``torch.is_grad_enabled()``. The teacher
        # forwards therefore already run unwrapped under ``torch.no_grad()``.
        if self.backend is None:
            raise RuntimeError("H3 training backend is not loaded")
        video = inputs.video.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.video is not None else None
        audio = inputs.audio.to(device=accelerator.device, dtype=self.dit_dtype) if inputs.audio is not None else None
        extension_kwargs = {}
        if self._step_row_video_timestep is not None:
            extension_kwargs["video_row_schedule"] = self._step_row_video_timestep
        if self._step_spatial_density_scale is not None:
            extension_kwargs["spatial_density_scale"] = self._step_spatial_density_scale
        anchors, anchor_indices = self._step_keyframes or ((), ())
        if anchors:
            extension_kwargs["condition_video_anchors"] = anchors
            extension_kwargs["extension_video_context"] = self._clean_latents(
                inputs.video, inputs.video_target, inputs.video_sigma
            ).index_select(-3, torch.tensor(anchor_indices, device=inputs.video.device))
        resolved_guides = self._step_guides or ()
        if resolved_guides:
            clean_video = self._clean_latents(inputs.video, inputs.video_target, inputs.video_sigma)
            clean_audio = (
                self._clean_latents(inputs.audio, inputs.audio_target, inputs.audio_sigma) if inputs.audio is not None else None
            )
            extension_kwargs["guide_geometries"] = tuple(item[0] for item in resolved_guides)
            extension_kwargs["guide_video_latents"] = tuple(
                clean_video[:, :, video_start : video_start + geometry.num_video_latents]
                for geometry, video_start, _ in resolved_guides
                if geometry.num_video_latents
            )
            extension_kwargs["guide_audio_latents"] = tuple(
                clean_audio[..., audio_start : audio_start + geometry.num_audio_latents]
                for geometry, _, audio_start in resolved_guides
                if geometry.num_audio_latents
            )
        if self._step_mask is not None:
            if self._step_mask.video_rows is not None:
                extension_kwargs["observed_video_rows"] = self._step_mask.video_rows
                extension_kwargs["clean_video_latents"] = self._clean_latents(inputs.video, inputs.video_target, inputs.video_sigma)
            if self._step_mask.audio_rows is not None:
                extension_kwargs["observed_audio_rows"] = self._step_mask.audio_rows
                extension_kwargs["clean_audio_latents"] = self._clean_latents(inputs.audio, inputs.audio_target, inputs.audio_sigma)
        # Zero on a step whose recipe draw did not select extension, which leaves
        # the packed layout identical to a run without the extension flags.
        extension_video_frames = self._active_extension_video_frames
        extension_audio_latents = self._active_extension_audio_latents
        if extension_video_frames or extension_audio_latents:
            extension_kwargs["extension_route"] = self._extension_route
        if extension_video_frames:
            extension_kwargs["extension_video_frames"] = extension_video_frames
            extension_kwargs["extension_video_context"] = self._clean_context(
                inputs.video, inputs.video_target, inputs.video_sigma, extension_video_frames, axis=-3
            )
        if extension_audio_latents:
            extension_kwargs["extension_audio_latents"] = extension_audio_latents
            extension_kwargs["extension_audio_context"] = self._clean_context(
                inputs.audio, inputs.audio_target, inputs.audio_sigma, extension_audio_latents, axis=-1
            )
        if self._step_reference_modality != "av":
            extension_kwargs["reference_modality"] = self._step_reference_modality
        # Read here, so the guidance empty branch and the base-preservation
        # teacher condition on the same presentation as the trainable branch --
        # the same sharing the mask, the keyframes and the modality variant use.
        if self._step_qwen_control_dropout:
            extension_kwargs["qwen_control_dropout"] = True
        return {
            "batch": batch,
            "video_hidden_states": video,
            "audio_hidden_states": audio,
            "video_timestep": inputs.video_timestep.to(accelerator.device),
            "audio_timestep": inputs.audio_timestep.to(accelerator.device),
            "conditioning": conditioning,
            # Forwarded only when extension is active so a backend that does not
            # implement it keeps its existing signature.
            **extension_kwargs,
        }

    def _predict(
        self,
        accelerator: Accelerator,
        transformer,
        batch,
        inputs,
        *,
        conditioning: str | tuple[str, ...],
        role: str | None = None,
    ) -> H3ModelPrediction:
        """One forward. ``role`` names it in the ``--h3_profile_steps`` scope table (``h3.forward.<role>``):
        by default a graph-carrying forward is the ``student`` and a no-grad one a ``teacher``."""
        call = self._predict_call(accelerator, batch, inputs, conditioning=conditioning)
        if role is None:
            role = "student" if torch.is_grad_enabled() else "teacher"
        # --h3_adapter_prompt_only: an empty-prompt forward is the frozen base's.
        # Left alone when the adapter is already off (a frozen bracket around
        # this call), so the restore below never re-enables it inside one.
        restore = None
        if getattr(self, "_adapter_prompt_only", False) and conditioning == "empty":
            network = getattr(self, "_adapter_network", None) or getattr(self, "_validation_network", None)
            if network is not None:
                unwrapped = accelerator.unwrap_model(network)
                # Ask the network when it can say (LoRA networks can); fall back to
                # the state the tracking toggle recorded, so a bracket made with the
                # raw set_enabled() is still respected wherever the network reports.
                is_enabled = getattr(unwrapped, "is_enabled", None)
                enabled = bool(is_enabled()) if callable(is_enabled) else getattr(unwrapped, "_h3_adapter_enabled", True)
                if enabled:
                    restore = self._runtime_network_toggle(accelerator, network, "--h3_adapter_prompt_only")
                    restore(False)
        try:
            with accelerator.autocast(), h3_profile_scope(f"h3.forward.{role}"):
                # Positional for the five arguments every backend has carried since
                # the first one, so a backend that named its parameters differently
                # keeps working.
                prediction = self.backend.predict_training(
                    transformer,
                    call["batch"],
                    call["video_hidden_states"],
                    call["audio_hidden_states"],
                    call["video_timestep"],
                    call["audio_timestep"],
                    **{key: value for key, value in call.items() if key not in _PREDICT_POSITIONAL_KEYS},
                )
        finally:
            if restore is not None:
                restore(True)
        if not isinstance(prediction, H3ModelPrediction):
            raise TypeError("H3 backend predict_training() must return H3ModelPrediction")
        return prediction

    def _predict_fused(self, accelerator: Accelerator, transformer, arms: list[H3FusedArm]) -> list[H3ModelPrediction]:
        """Run several described forwards through one pass over the blocks."""
        if self.backend is None:
            raise RuntimeError("H3 training backend is not loaded")
        fused = getattr(self.backend, "predict_training_fused", None)
        if not callable(fused):
            raise TypeError(
                "--h3_rollout_fused_teacher requires a training backend implementing predict_training_fused(); "
                "drop the flag to use the ordinary two-forward path"
            )
        with accelerator.autocast(), h3_profile_scope("h3.forward.fused"):
            predictions = fused(transformer, arms)
        if len(predictions) != len(arms) or not all(isinstance(item, H3ModelPrediction) for item in predictions):
            raise TypeError("H3 backend predict_training_fused() must return one H3ModelPrediction per arm")
        return list(predictions)

    @staticmethod
    def _split_paired_prediction(prediction: H3ModelPrediction) -> tuple[H3ModelPrediction, H3ModelPrediction]:
        present = [value for value in (prediction.video, prediction.audio) if value is not None]
        if not present or any(value.shape[0] != 2 for value in present):
            raise TypeError("H3 paired teacher prediction must contain a batch of two for every present modality")
        return tuple(
            H3ModelPrediction(
                None if prediction.video is None else prediction.video[index : index + 1],
                None if prediction.audio is None else prediction.audio[index : index + 1],
            )
            for index in range(2)
        )

    def get_primary_latents(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "latents" in batch:
            return batch["latents"]
        if H3_AUDIO_LATENTS_KEY in batch:
            return batch[H3_AUDIO_LATENTS_KEY]
        raise KeyError("MiniMax H3 cache contains neither video nor audio target latents")

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self._batch_backward_performed = False
        self._adapter_network = network
        self._adapter_prompt_only = bool(getattr(args, "h3_adapter_prompt_only", False))
        # The block-swap arm gate is balanced by construction -- one hook firing
        # per announced invocation -- but a step that raises between forward and
        # backward leaves announcements nothing will consume. An H2D-only ring
        # never re-prepares mid-training, so without this its gate would stay
        # poisoned for the rest of the run. The previous step's backward is done
        # by the time we get here, so nothing pending is ever discarded.
        reset_backward_arms = getattr(getattr(transformer, "offloader", None), "reset_backward_arms", None)
        if callable(reset_backward_arms):
            reset_backward_arms()
        batch_size = int(latents.shape[0])
        if batch_size < 1:
            raise ValueError("MiniMax H3 training received an empty batch")
        preservation_active = args.h3_base_preservation_loss_weight > 0 and self._base_preservation_active(
            accelerator, args.h3_base_preservation_probability
        )
        dop_active = args.h3_dop_loss_weight > 0 and self._dop_probability_active(accelerator, args.h3_dop_probability)
        guidance_active = args.h3_guidance_distillation_scale is not None and self._guidance_distillation_active(
            accelerator, args.h3_guidance_distillation_probability
        )
        # One recipe per optimizer step: every item of a batch trains the same
        # conditioning objective, exactly as the preservation and guidance draws
        # above are shared. Its stream is independent of theirs, so the draw order
        # here does not couple the three.
        recipe = self._draw_step_recipe(accelerator, args)
        # One control-dropout decision per step, on its own stream, for the same
        # reason: every item of a batch trains the same presentation.
        qwen_control_dropout = args.h3_qwen_control_dropout_rate > 0 and self._qwen_control_dropout_active(
            accelerator, args.h3_qwen_control_dropout_rate
        )
        # One rollout decision per optimizer step, on its own stream, so every
        # item of a batch optimizes the same objective -- the same contract the
        # guidance, preservation and recipe draws follow.
        rollout_active = getattr(args, "h3_rollout_supervision", False) and self._rollout_supervision_active(
            accelerator, args.h3_rollout_probability
        )
        if batch_size == 1:
            return self._process_single_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
                preservation_active_override=preservation_active,
                dop_active_override=dop_active,
                auxiliary_backward_scale=1.0,
                guidance_active_override=guidance_active,
                recipe_override=recipe,
                qwen_control_dropout_override=qwen_control_dropout,
                rollout_active_override=rollout_active,
            )

        # The released H3 transformer accepts one shared packed layout, while
        # prompts, references and task presentations are variable-length. Run
        # each packed item independently and backpropagate its scaled loss
        # immediately, so padding cannot leak through attention and only one
        # block-swap/checkpoint graph is alive at a time.
        losses: list[torch.Tensor] = []
        item_metrics: list[dict[str, float]] = []
        crepa_alignments: list[float] = []
        for index in range(batch_size):
            # DDP decides whether to synchronize while its forward hooks run,
            # so no_sync must cover both the forward and matching backward.
            sync_context = (
                accelerator.no_sync(network if network is not None else transformer)
                if index + 1 < batch_size and getattr(accelerator, "num_processes", 1) > 1
                else nullcontext()
            )
            with sync_context:
                item_loss, metrics = self._process_single_batch(
                    args,
                    accelerator,
                    transformer,
                    network,
                    self._slice_batch_item(batch, index, batch_size),
                    latents[index : index + 1],
                    noise[index : index + 1],
                    noise_scheduler,
                    dit_dtype,
                    network_dtype,
                    vae,
                    global_step,
                    preservation_active_override=preservation_active,
                    dop_active_override=dop_active,
                    auxiliary_backward_scale=1.0 / batch_size,
                    guidance_active_override=guidance_active,
                    recipe_override=recipe,
                    qwen_control_dropout_override=qwen_control_dropout,
                    rollout_active_override=rollout_active,
                    crepa_update_similarity_threshold=False,
                )
                accelerator.backward(item_loss / batch_size)
            if "crepa/alignment" in metrics:
                crepa_alignments.append(metrics["crepa/alignment"])
            losses.append(item_loss.detach())
            item_metrics.append(metrics)

        averaged_metrics = self._average_batch_metrics(item_metrics)
        if self._crepa is not None and crepa_alignments:
            self._crepa.update_similarity_threshold(sum(crepa_alignments) / len(crepa_alignments))
            averaged_metrics["crepa/cutoff"] = float(self._crepa._cutoff_active)
            if self._crepa._similarity_ema is not None:
                averaged_metrics["crepa/alignment_ema"] = self._crepa._similarity_ema
        self._batch_backward_performed = True
        return torch.stack(losses).mean(), averaged_metrics

    def backward_loss(self, accelerator: Accelerator, loss: torch.Tensor) -> None:
        try:
            if getattr(self, "_batch_backward_performed", False):
                self._batch_backward_performed = False
                return
            super().backward_loss(accelerator, loss)
        finally:
            # Checkpoint recomputation needs the hook capture through backward,
            # but keeping it until the next step wastes the exact headroom used
            # by validation and sampling between optimizer steps.
            if self._crepa is not None:
                self._crepa.clear_step()

    @staticmethod
    def _average_batch_metrics(item_metrics: list[dict[str, float]]) -> dict[str, float]:
        metric_keys = set().union(*(metrics.keys() for metrics in item_metrics))
        return {
            key: sum(metrics[key] for metrics in item_metrics if key in metrics) / sum(key in metrics for metrics in item_metrics)
            for key in metric_keys
        }

    @staticmethod
    def _slice_batch_item(batch: dict, index: int, batch_size: int) -> dict:
        # BucketBatchManager stacks every fixed-size cache field on its first
        # dimension and leaves every varlen_ field as a list. Keep this rule in
        # one place; new shared metadata must remain scalar or opt into one of
        # those two dataset representations.
        item = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                item[key] = value[index : index + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                item[key] = [value[index]]
            elif isinstance(value, tuple) and len(value) == batch_size:
                item[key] = (value[index],)
            else:
                item[key] = value
        return item

    @staticmethod
    def _target_has_valid_elements(batch: dict, key: str, present: bool) -> bool:
        if not present:
            return False
        mask = batch.get(key)
        if mask is None:
            return True
        if isinstance(mask, (list, tuple)):
            if len(mask) != 1:
                raise ValueError(f"H3 {key} must contain one batch item")
            mask = mask[0]
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"H3 {key} must be a tensor")
        return bool(mask.any())

    def _process_single_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
        *,
        preservation_active_override: bool | None = None,
        dop_active_override: bool | None = None,
        auxiliary_backward_scale: float = 1.0,
        guidance_active_override: bool | None = None,
        recipe_override: str | None = None,
        qwen_control_dropout_override: bool | None = None,
        rollout_active_override: bool | None = None,
        crepa_update_similarity_threshold: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del network_dtype, vae
        # EXPERIMENTAL: one presentation per step. A dropped step feeds the
        # control-free twin cached beside the control presentation to every
        # branch; no loss is rescaled, exactly as caption dropout does not.
        if qwen_control_dropout_override is None:
            self._step_qwen_control_dropout = args.h3_qwen_control_dropout_rate > 0 and self._qwen_control_dropout_active(
                accelerator, args.h3_qwen_control_dropout_rate
            )
        else:
            self._step_qwen_control_dropout = qwen_control_dropout_override
        self._step_reference_modality = "av"
        probabilities = batch.get(H3_REFERENCE_MODALITY_PROBABILITIES_KEY)
        if probabilities is not None:
            if isinstance(probabilities, (list, tuple)):
                if len(probabilities) != 1:
                    raise ValueError("H3 reference modality probabilities must contain one batch item")
                probabilities = probabilities[0]
            if probabilities.ndim == 2 and probabilities.shape[0] == 1:
                probabilities = probabilities[0]
            probabilities = probabilities.detach().to(device="cpu", dtype=torch.float32)
            if probabilities.shape != (3,):
                raise ValueError("H3 reference modality probabilities must have shape [3]")
            selected = int(torch.multinomial(probabilities, 1).item())
            self._step_reference_modality = ("av", "video", "audio")[selected]
        has_video = "latents" in batch or latents.ndim == 5
        has_audio = H3_AUDIO_LATENTS_KEY in batch
        if not has_video and not has_audio:
            raise KeyError("MiniMax H3 cache contains no target modality")
        video_source = batch.get("latents", latents if latents.ndim == 5 else None)
        video_latents = video_source.to(device=accelerator.device, dtype=dit_dtype) if has_video else None
        audio_latents = batch[H3_AUDIO_LATENTS_KEY].to(device=accelerator.device, dtype=dit_dtype) if has_audio else None
        spatial_tokens = bool(args.h3_audio_only_spatial_tokens and not has_video and has_audio)
        if spatial_tokens:
            video_latents = self._build_audio_only_spatial_tokens(audio_latents)
        video_noise = (
            torch.randn_like(video_latents)
            if spatial_tokens
            else noise.to(device=accelerator.device, dtype=dit_dtype)
            if video_latents is not None
            else None
        )
        audio_noise = (
            noise.to(device=accelerator.device, dtype=dit_dtype)
            if audio_latents is not None and not has_video
            else torch.randn_like(audio_latents)
            if audio_latents is not None
            else None
        )
        is_image = has_video and not has_audio and video_latents.shape[2] == 1

        observed = args.h3_observed_modality
        if observed == "random":
            # One adapter covering joint generation, audio-driven video and
            # video-to-audio: the task is redrawn per step rather than fixed for
            # the run, so the model keeps all three rather than specialising.
            # Do not select a direction whose generated side is fully masked
            # (most commonly video-to-audio on a silent clip), because that
            # would spend a complete optimizer step on an exactly zero loss.
            valid_video = self._target_has_valid_elements(batch, "video_loss_mask", has_video)
            valid_audio = self._target_has_valid_elements(batch, "audio_loss_mask", has_audio)
            candidates = []
            if (valid_video and args.h3_video_loss_weight > 0) or (valid_audio and args.h3_audio_loss_weight > 0):
                candidates.append(None)
            if has_video and has_audio and valid_audio and args.h3_audio_loss_weight > 0:
                candidates.append("video")
            if has_video and has_audio and valid_video and args.h3_video_loss_weight > 0:
                candidates.append("audio")
            observed = candidates[int(torch.randint(0, len(candidates), (1,), device="cpu").item())] if candidates else None
        if observed is not None and not (has_video and has_audio):
            present = "video" if has_video else "audio" if has_audio else "neither"
            raise ValueError(
                f"--h3_observed_modality reads one modality while training the other, so batches must "
                f"carry both; this batch carries {present}. Cache the dataset with target_modalities = ['video', 'audio']."
            )

        scheduler_args = args
        if is_image:
            patch_h, patch_w = VIDEO_DIT_PATCH_SIZE[-2:]
            latent_height, latent_width = video_latents.shape[-2:]
            if latent_height % patch_h or latent_width % patch_w:
                raise ValueError("MiniMax H3 image latent dimensions must be divisible by the spatial patch size")
            scheduler_args = copy.copy(args)
            if args.h3_image_flow_shift is None:
                # Use the common logit-normal density with a
                # resolution-aware shift for image batches.
                scheduler_args.timestep_sampling = "krea2_shift"
            else:
                scheduler_args.timestep_sampling = "shift"
                scheduler_args.discrete_flow_shift = args.h3_image_flow_shift

        _, scheduler_timesteps = super().get_noisy_model_input_and_timesteps(
            scheduler_args,
            noise,
            latents,
            batch["timesteps"],
            noise_scheduler,
            accelerator.device,
            dit_dtype,
            return_noisy=False,
        )
        base_sigma = self._base_sigma(scheduler_args, noise_scheduler, scheduler_timesteps, accelerator.device)
        if not is_image:
            base_sigma = _apply_timestep_focus(
                base_sigma,
                args.h3_timestep_focus_min,
                args.h3_timestep_focus_max,
                args.h3_timestep_focus_probability,
            )
        inputs = prepare_joint_noisy_inputs(
            video_latents,
            audio_latents,
            video_noise,
            audio_noise,
            base_sigma,
            # Image sampling already returned its final shifted sigma. Video
            # batches instead receive H3's synchronized 12/3 shifts here.
            video_shift=1.0 if is_image else args.h3_shift_video,
            audio_shift=1.0 if is_image else args.h3_shift_audio,
            observed=observed,
        )

        inputs, self._step_row_video_timestep = self._apply_frame_sigma_jitter(
            args, inputs, video_latents, video_noise, base_sigma, is_image
        )
        self._step_spatial_density_scale = self._draw_spatial_density_scale()
        # Bind this step's conditioning recipe before anything reads it. ``None``
        # means no mixing is configured, so the mask draw and the extension
        # context below behave exactly as they did before recipe mixing existed.
        self._step_recipe = recipe_override if recipe_override is not None else self._draw_step_recipe(accelerator, args)
        self._step_mask = self._draw_step_mask(inputs, tuple(VIDEO_DIT_PATCH_SIZE), batch)
        # Drawn once per step for the same reason the mask is: the guidance and
        # base-preservation branches must condition on the same anchors as the
        # trainable branch, or the guidance correction inverts a different field.
        self._step_keyframes = self._resolve_keyframe_anchors(inputs.video)
        self._step_guides = self._resolve_guide_specs(inputs.video, inputs.audio)

        # H3 trains one item per step, so caption dropout is a single draw rather
        # than a per-sample mask. A dropped step trains the unconditional branch,
        # which is what gives the adapter a null prompt to contrast against at
        # inference; without it the model has never seen one.
        conditioning = "prompt"
        if args.h3_caption_dropout_rate > 0:
            missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
            if missing_empty:
                raise KeyError("--h3_caption_dropout_rate requires --cache_guidance_empty; missing " + ", ".join(missing_empty))
            if float(torch.rand((), device="cpu")) < args.h3_caption_dropout_rate:
                conditioning = "empty"

        # A dropped step is already unconditional, so there is no guided field to
        # invert and both branches would evaluate the same empty prompt.
        use_guidance = args.h3_guidance_distillation_scale is not None and conditioning == "prompt"
        # Sparse guidance skips the empty forward entirely on an inactive step and
        # falls back to the ordinary velocity objective for that step.
        if use_guidance:
            use_guidance = (
                self._guidance_distillation_active(accelerator, args.h3_guidance_distillation_probability)
                if guidance_active_override is None
                else guidance_active_override
            )
        # The teacher's privilege lives entirely in its prompted presentation, so
        # there is no privileged null field to distil into a dropped step's
        # unconditional branch. Such a step keeps the ordinary data objective,
        # exactly as the guidance branch stands down on one.
        rollout_active = bool(getattr(args, "h3_rollout_supervision", False)) and conditioning == "prompt"
        if rollout_active:
            rollout_active = (
                self._rollout_supervision_active(accelerator, args.h3_rollout_probability)
                if rollout_active_override is None
                else rollout_active_override
            )
        reference_prediction = None
        null_anchor_student = None
        null_anchor_reference = None
        null_anchor = None
        preservation_active = args.h3_base_preservation_loss_weight > 0 and (
            self._base_preservation_active(accelerator, args.h3_base_preservation_probability)
            if preservation_active_override is None
            else preservation_active_override
        )
        dop_active = (
            args.h3_dop_loss_weight > 0
            and conditioning == "prompt"
            and (
                self._dop_probability_active(accelerator, args.h3_dop_probability)
                if dop_active_override is None
                else dop_active_override
            )
        )
        dop_reference_prediction = None
        if dop_active:
            missing_dop = [
                key for key in (H3_DOP_TEXT_HIDDEN_KEY, H3_DOP_TEXT_TOKEN_TAGS_KEY, H3_DOP_CONFIG_KEY) if key not in batch
            ]
            if missing_dop:
                raise KeyError(
                    "H3 DOP requires text caches written with --h3_dop_trigger and --h3_dop_class_prompt; missing "
                    + ", ".join(missing_dop)
                )
            cached_identity = batch[H3_DOP_CONFIG_KEY]
            if isinstance(cached_identity, (list, tuple)):
                cached_identity = cached_identity[0]
            cached_identity = cached_identity.reshape(-1).cpu()
            if not torch.equal(cached_identity, dop_config_identity(args.h3_dop_trigger, args.h3_dop_class_prompt)):
                raise ValueError("H3 DOP cache identity does not match the requested trigger/class pair; re-cache conditioning")
        # H2D-only LoRA rings self-heal at same-direction forward boundaries.
        # Classic swap (and the dense trainable ring) instead expects backward
        # to restore the training layout and must explicitly enter forward-only
        # mode around no-grad teacher passes.
        # The null anchor's frozen reference forward is a no-grad teacher pass
        # like the others and needs the same forward-only bracket; without it a
        # run with nothing but the anchor left classic swap in the training
        # layout after a forward that never ran its backward.
        null_anchor_probability = _anchor_probability(args)
        null_anchor_active = (
            float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0) > 0
            and conditioning == "prompt"
            and self._null_anchor_probability_active(accelerator, null_anchor_probability)
        )
        auxiliary_block_swap = (
            bool(self.blocks_to_swap)
            and not getattr(self, "_block_swap_h2d_only", False)
            and (use_guidance or preservation_active or dop_active or null_anchor_active)
        )
        if auxiliary_block_swap:
            # Teacher branches have no backward pass. Classic block swap in
            # training mode leaves the forward prefix on CPU for its backward
            # hooks to restore, so every auxiliary forward must use the cyclic
            # forward-only schedule and return to the training layout before
            # the graph-carrying student forward.
            transformer.switch_block_swap_for_inference()
        try:
            fused_teachers = bool(
                args.h3_fuse_frozen_teachers
                and use_guidance
                and preservation_active
                and args.h3_guidance_null_source == "frozen"
                and getattr(self.backend, "supports_paired_conditioning", False)
            )
            fused_teachers_completed = False
            if fused_teachers:
                missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
                if missing_empty:
                    raise KeyError(
                        "guidance-consistent H3 training requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
                    )
                fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_fuse_frozen_teachers")
                int8_context = getattr(transformer, "int8_attention_context", None)
                with (
                    torch.random.fork_rng(devices=fork_devices),
                    torch.no_grad(),
                    int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
                ):
                    set_enabled(False)
                    try:
                        try:
                            paired_prediction = self._predict(
                                accelerator,
                                transformer,
                                batch,
                                inputs,
                                conditioning=("empty", conditioning),
                            )
                        except H3PairedConditioningUnsupportedError as error:
                            if not getattr(self, "_paired_teacher_fallback_warned", False):
                                logger.warning("%s", error)
                                self._paired_teacher_fallback_warned = True
                        else:
                            empty_prediction, reference_prediction = self._split_paired_prediction(paired_prediction)
                            fused_teachers_completed = True
                    finally:
                        set_enabled(True)

            if use_guidance and not fused_teachers_completed:
                missing_empty = [key for key in (H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY) if key not in batch]
                if missing_empty:
                    raise KeyError(
                        "guidance-consistent H3 training requires --cache_guidance_empty; missing " + ", ".join(missing_empty)
                    )
                # The empty branch calibrates the distilled field but is not itself
                # optimized. Evaluate it first without retaining its autograd graph.
                fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                int8_context = getattr(transformer, "int8_attention_context", None)
                # A frozen null branch is a fixed anchor: the adapter is disabled
                # for this forward only, so the field the guidance correction
                # inverts cannot drift along with the adapter that is being
                # trained against it.
                null_set_enabled = (
                    self._runtime_network_toggle(accelerator, network, "--h3_guidance_null_source frozen")
                    if args.h3_guidance_null_source == "frozen"
                    else None
                )
                with (
                    torch.random.fork_rng(devices=fork_devices),
                    torch.no_grad(),
                    int8_context(auxiliary=True) if callable(int8_context) else nullcontext(),
                ):
                    if null_set_enabled is not None:
                        null_set_enabled(False)
                    try:
                        empty_prediction = self._predict(
                            accelerator,
                            transformer,
                            batch,
                            inputs,
                            conditioning="empty",
                        )
                    finally:
                        if null_set_enabled is not None:
                            null_set_enabled(True)
            null_anchor_weight = float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0)
            if null_anchor_active:
                # Hold the EMPTY-prompt prediction where the checkpoint had it, and
                # leave the prompted one entirely free.
                #
                # Why the empty branch and not the prompted one: preserving the
                # PROMPTED branch fights the data term directly -- it says "do not change
                # your answer to the prompt", which is the thing training is for --
                # whereas the empty branch is a degree of freedom that learning a concept
                # does not need, so constraining it is nearly free of that conflict.
                #
                # What the evidence does and does not say. It was adopted on the strength
                # of a field-distance metric built from (prompted - empty), which this
                # term directly holds still; that reading was circular and is withdrawn.
                # What survives is measured where the empty branch cancels out of the
                # arithmetic -- prompted-branch drift against the checkpoint -- where the
                # anchored arm sits about 0.03 closer across five matched points, and a
                # six-pair blind render comparison at matched learning that went 6-0.
                # One seed, small corpora.
                #
                # Costs one grad forward and one no-grad forward, about 1.8x a step.
                # Skipped on a caption-dropout step, where the data objective is already
                # training the empty branch toward the clip and two instructions would
                # be aimed at one prediction.
                null_anchor_toggle = self._runtime_network_toggle(accelerator, network, "--h3_guidance_null_anchor_weight")
                anchor_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                anchor_int8 = getattr(transformer, "int8_attention_context", None)
                with self._trainable_block_swap(transformer, auxiliary_block_swap):
                    null_anchor_student = self._predict(accelerator, transformer, batch, inputs, conditioning="empty")
                with (
                    torch.random.fork_rng(devices=anchor_devices),
                    torch.no_grad(),
                    anchor_int8(auxiliary=True) if callable(anchor_int8) else nullcontext(),
                ):
                    null_anchor_toggle(False)
                    try:
                        null_anchor_reference = self._predict(accelerator, transformer, batch, inputs, conditioning="empty")
                    finally:
                        null_anchor_toggle(True)
            if preservation_active and not fused_teachers_completed:
                set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_base_preservation_loss_weight")
                fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                # Restoring the RNG state makes the following trainable pass reuse
                # the stochastic conditioning rows sampled by the frozen branch.
                with torch.random.fork_rng(devices=fork_devices):
                    set_enabled(False)
                    try:
                        int8_context = getattr(transformer, "int8_attention_context", None)
                        with torch.no_grad(), int8_context(auxiliary=True) if callable(int8_context) else nullcontext():
                            reference_prediction = self._predict(
                                accelerator,
                                transformer,
                                batch,
                                inputs,
                                # Match the student's conditioning, or a dropped step
                                # would pull the unconditional branch toward the
                                # frozen base's conditional prediction.
                                conditioning=conditioning,
                            )
                    finally:
                        set_enabled(True)
            if dop_active:
                set_enabled = self._runtime_network_toggle(accelerator, network, "--h3_dop_loss_weight")
                fork_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    set_enabled(False)
                    try:
                        int8_context = getattr(transformer, "int8_attention_context", None)
                        with torch.no_grad(), int8_context(auxiliary=True) if callable(int8_context) else nullcontext():
                            dop_reference_prediction = self._predict(accelerator, transformer, batch, inputs, conditioning="dop")
                    finally:
                        set_enabled(True)
        finally:
            if auxiliary_block_swap:
                transformer.switch_block_swap_for_training()

        dop_term = None
        if dop_reference_prediction is not None:
            # Backpropagate this auxiliary graph before constructing the primary
            # graph. This keeps classic block swapping valid and bounds activation
            # memory even when the rewritten prompt has a different packed length.
            dop_prediction = self._predict(accelerator, transformer, batch, inputs, conditioning="dop")
            dop_video_mask = self._mask_to_loss(
                self._extension_masked(
                    batch.get("video_loss_mask"), inputs.video_target, self._active_extension_video_frames, axis=-3
                ),
                inputs.video_target,
                None if self._step_mask is None else self._step_mask.video_latent,
                axis=-3,
            )
            dop_audio_mask = self._mask_to_loss(
                self._extension_masked(
                    batch.get("audio_loss_mask"), inputs.audio_target, self._active_extension_audio_latents, axis=-1
                ),
                inputs.audio_target,
                None if self._step_mask is None else self._step_mask.audio_latent,
                axis=-1,
            )
            dop_preservation = joint_prediction_loss(
                dop_prediction,
                dop_reference_prediction,
                video_mask=dop_video_mask,
                audio_mask=dop_audio_mask,
                video_sample_weight=self._sample_weight(args, inputs.video_sigma) if has_video else None,
                audio_sample_weight=self._sample_weight(args, inputs.audio_sigma, modality="audio") if has_audio else None,
                balance=args.h3_loss_balance,
                mask_normalization=args.h3_loss_mask_normalization,
                video_weight=0.0 if observed == "video" or spatial_tokens else args.h3_video_loss_weight,
                audio_weight=0.0 if observed == "audio" else args.h3_audio_loss_weight,
            )
            dop_term = (args.h3_dop_loss_weight / args.h3_dop_probability) * dop_preservation.loss
            accelerator.backward(dop_term * auxiliary_backward_scale)
            del dop_prediction

        use_crepa = self._crepa is not None and has_video and not is_image and observed != "video"
        if self._crepa is not None:
            self._crepa.begin_step(use_crepa, global_step)
        try:
            raw_prediction = self._predict(
                accelerator,
                transformer,
                batch,
                inputs,
                conditioning=conditioning,
            )
        except Exception:
            if self._crepa is not None:
                self._crepa.clear_step()
            raise
        prediction = raw_prediction
        loss_inputs = inputs
        if use_guidance:
            prediction, loss_inputs = self._guidance_loss_inputs(
                args, prediction, empty_prediction, inputs, accelerator=accelerator
            )

        video_sample_weight = self._sample_weight(args, inputs.video_sigma) if has_video else None
        audio_sample_weight = self._sample_weight(args, inputs.audio_sigma, modality="audio") if has_audio else None
        video_weight = 0.0 if observed == "video" or spatial_tokens else args.h3_video_loss_weight
        audio_weight = 0.0 if observed == "audio" else args.h3_audio_loss_weight
        effective_video_mask = self._mask_to_loss(
            self._extension_masked(batch.get("video_loss_mask"), inputs.video_target, self._active_extension_video_frames, axis=-3),
            inputs.video_target,
            None if self._step_mask is None else self._step_mask.video_latent,
            axis=-3,
        )
        effective_audio_mask = self._mask_to_loss(
            self._extension_masked(
                batch.get("audio_loss_mask"), inputs.audio_target, self._active_extension_audio_latents, axis=-1
            ),
            inputs.audio_target,
            None if self._step_mask is None else self._step_mask.audio_latent,
            axis=-1,
        )

        def _velocity_loss(step_prediction, step_inputs):
            return joint_velocity_loss(
                step_prediction,
                step_inputs,
                # The observed context is given, not predicted, so it carries no
                # training signal and would otherwise dominate a short continuation.
                video_mask=effective_video_mask,
                audio_mask=effective_audio_mask,
                # Weighting keys on the shifted sigma the model actually saw for the
                # modality being generated, not the shared unshifted coordinate. An
                # observed modality sits at a pinned constant and would carry no
                # schedule information.
                video_sample_weight=video_sample_weight,
                audio_sample_weight=audio_sample_weight,
                balance=args.h3_loss_balance,
                mask_normalization=args.h3_loss_mask_normalization,
                # The observed modality is conditioning, not a target.
                video_weight=video_weight,
                audio_weight=audio_weight,
            )

        result = _velocity_loss(prediction, loss_inputs)
        # A modality with weight 0 (the observed side of a v2a/a2v step) reports a
        # flat 0.0 rather than disappearing: the key stays in every step's metric
        # set so existing dashboards and the per-item averaging below keep a
        # constant schema.
        metrics = {
            "loss/video": float(result.video_loss.detach()),
            "loss/audio": float(result.audio_loss.detach()),
            "h3/sigma_video": float(inputs.video_sigma.mean().detach()),
            "h3/sigma_audio": float(inputs.audio_sigma.mean().detach()),
        }
        if args.h3_dop_loss_weight > 0:
            metrics["h3/dop_active"] = float(dop_active)
            metrics["loss/dop"] = 0.0 if dop_term is None else float(dop_term.detach())
        if result.video_elements == 0 and result.audio_elements == 0:
            metrics["h3/no_active_target"] = 1.0
        if args.h3_caption_dropout_rate > 0:
            # Only reported when the feature is on, so an existing run's metric
            # set is unchanged.
            metrics["h3/caption_dropped"] = float(conditioning == "empty")
        if args.h3_qwen_control_dropout_rate > 0:
            # Only reported when the feature is on, so an existing run's metric
            # set is unchanged.
            metrics["h3/qwen_control_dropout_active"] = float(self._step_qwen_control_dropout)
        if self._step_recipe is not None:
            # Only reported once a probability below 1 turns mixing on, so a
            # single-recipe run's metric set is unchanged. Recipe mixing trains a
            # different objective on the selected steps rather than a sparse
            # estimate of one objective, so no loss is rescaled by its probability.
            mask_configured, extension_configured = self._configured_recipes()
            if mask_configured:
                metrics["h3/recipe_mask_active"] = float(self._step_recipe == "mask")
            if extension_configured:
                metrics["h3/recipe_extension_active"] = float(self._step_recipe == "extension")
        loss = result.loss
        # Dense-equivalent objective, free of inverse-probability scaling and of
        # auxiliary terms. Reported as the averaged loss whenever the optimized
        # loss differs from it.
        dense_loss = result.loss
        rollout_replaced = False
        rollout_field_floor = None
        rollout_null_anchor = None
        if rollout_active:
            # On-policy supervision REPLACES the video half of the data objective
            # and leaves the audio half alone. The two halves are reduced in one
            # joint call per sub-step rather than separately, so the mask, the
            # per-modality weights and the token/modality balance are the ones
            # every other H3 loss uses. Averaging that joint loss over the window
            # is exactly "the audio data loss plus the averaged rollout video
            # loss": the audio term is identical across sub-steps and both
            # balances are linear in the video numerator at a fixed denominator.
            rollout_pairs, rollout_stop_sigma, rollout_video_sigmas, rollout_bases, rollout_student_empties = (
                self._rollout_supervision(
                    args,
                    accelerator,
                    transformer,
                    network,
                    batch,
                    video_latents=video_latents,
                    audio_latents=audio_latents,
                    video_shift=1.0 if is_image else args.h3_shift_video,
                    audio_shift=1.0 if is_image else args.h3_shift_audio,
                    conditioning=conditioning,
                    auxiliary_block_swap=bool(self.blocks_to_swap) and not getattr(self, "_block_swap_h2d_only", False),
                )
            )
            # The audio half is the data-forward objective this step would have run
            # anyway -- including its guidance modification when the hybrid is on.
            # ``prediction``/``loss_inputs`` are exactly what ``_guidance_loss_inputs``
            # produced, so the normalized form (which rewrites the prediction) and
            # the contrastive form (which rewrites the target) both land here
            # unchanged, and with guidance off they are ``raw_prediction``/``inputs``
            # and the term is bit-for-bit the plain one. No extra forward is paid:
            # the empty branch the guidance needs was already evaluated at the data
            # state above.
            rollout_audio_prediction = prediction.audio
            rollout_audio_target = loss_inputs.audio_target
            rollout_terms = [
                joint_prediction_loss(
                    H3ModelPrediction(video=student.video, audio=rollout_audio_prediction),
                    # ``joint_prediction_loss`` detaches its whole reference, which
                    # is the stop-gradient on the teacher; detaching the audio data
                    # target alongside it is a no-op.
                    H3ModelPrediction(video=teacher.video, audio=rollout_audio_target),
                    # The authored masks describe padding and authored regions of
                    # the packed layout, which a rollout state shares with the data
                    # state because it has the same shape; keeping them out would
                    # train the video field on padding rows.
                    video_mask=effective_video_mask,
                    audio_mask=effective_audio_mask,
                    # The video half is scored at the sub-step's own shifted sigma;
                    # the audio half is still the data step and keeps the data
                    # step's weight.
                    video_sample_weight=self._sample_weight(args, sigma) if has_video else None,
                    audio_sample_weight=audio_sample_weight,
                    balance=args.h3_loss_balance,
                    mask_normalization=args.h3_loss_mask_normalization,
                    video_weight=video_weight,
                    audio_weight=audio_weight,
                )
                for (student, teacher), sigma in zip(rollout_pairs, rollout_video_sigmas, strict=True)
            ]
            window = float(len(rollout_terms))
            loss = sum(term.loss for term in rollout_terms) / window
            rescaled_rollout_loss = loss
            rollout_replaced = True
            metrics["loss/rollout_video"] = float(sum(term.video_loss.detach() for term in rollout_terms) / window)
            metrics["h3/rollout_stop_sigma"] = rollout_stop_sigma
            if rollout_student_empties is not None:
                # The null anchor at the supervised states: the student's own empty
                # branch held to the frozen base's, where generation walks.
                anchor_terms = [
                    joint_prediction_loss(
                        student_empty,
                        base_empty,
                        video_mask=effective_video_mask,
                        audio_mask=effective_audio_mask,
                        video_sample_weight=self._sample_weight(args, sigma) if has_video else None,
                        audio_sample_weight=audio_sample_weight,
                        balance=args.h3_loss_balance,
                        mask_normalization=args.h3_loss_mask_normalization,
                        video_weight=video_weight,
                        audio_weight=audio_weight,
                    ).loss
                    for student_empty, (_base_prompted, base_empty), sigma in zip(
                        rollout_student_empties, rollout_bases, rollout_video_sigmas, strict=True
                    )
                ]
                rollout_null_anchor = float(args.h3_rollout_null_anchor_weight) * sum(anchor_terms) / float(len(anchor_terms))
                loss = loss + rollout_null_anchor
                metrics["loss/rollout_null_anchor"] = float(rollout_null_anchor.detach())
            if rollout_bases is not None and float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0) > 0:
                # A LENGTH floor on the guidance field at the supervised states.
                # The teacher term above sets the field's direction; this holds
                # its length at no less than the checkpoint's, and nothing more.
                # Both fields are taken against the FROZEN empty branch: with the
                # student's own null in the difference, the cheapest way to lengthen
                # the field would be to move that null, which is the degeneracy the
                # frozen-null forms exist to remove.
                # Along the student's own field ('self'), or along the teacher's
                # field at the same state ('teacher'), where lengthening in a wrong
                # direction earns nothing and the gradient turns the field as it
                # lengthens it.
                along_teacher = getattr(args, "h3_rollout_field_floor_direction", "self") == "teacher"
                measured = []
                for (student, teacher), (base_prompted, base_empty) in zip(rollout_pairs, rollout_bases, strict=True):
                    empty = base_empty.video.detach()
                    adapted_field = student.video - empty
                    base_field = base_prompted.video - empty
                    if along_teacher:
                        measured.append(
                            self._field_projection_ratio(adapted_field, base_field, teacher.video - empty, effective_video_mask)
                        )
                    else:
                        measured.append(self._field_length_ratio(adapted_field, base_field, effective_video_mask))
                ratios = torch.stack([ratio for ratio, _valid in measured])
                valid = torch.stack([flag for _ratio, flag in measured]).to(ratios.dtype)
                # States above the sigma ceiling drop out of the mean, like
                # unauthored samples.
                sigma_max = float(getattr(args, "h3_rollout_field_floor_sigma_max", 1.0))
                if sigma_max < 1.0:
                    in_band = torch.stack(
                        [(sigma.reshape(-1)[:1] <= sigma_max).to(ratios.dtype).to(ratios.device) for sigma in rollout_video_sigmas]
                    ).expand_as(valid)
                    valid = valid * in_band
                scored = valid.sum().clamp_min(1.0)
                band = torch.relu(1.0 - ratios).pow(2)
                field_cap = float(getattr(args, "h3_rollout_field_cap", 0.0) or 0.0)
                if field_cap > 0:
                    # Two-sided: an over-long field burns colour and, once
                    # co-adapted into the weights, is not undone by an inference
                    # multiplier, so it is held from above as well.
                    band = band + torch.relu(ratios - field_cap).pow(2)
                rollout_field_floor = float(args.h3_rollout_field_floor) * (band * valid).sum() / scored
                loss = loss + rollout_field_floor
                metrics["loss/rollout_field_floor"] = float(rollout_field_floor.detach())
                metrics["h3/rollout_field_ratio"] = float((ratios.detach() * valid).sum() / scored)
        if getattr(args, "h3_rollout_supervision", False):
            # Only reported when the feature is on, so an existing run's metric
            # set is unchanged.
            metrics["h3/rollout_active"] = float(rollout_active)
            metrics.setdefault("loss/rollout_video", 0.0)
            if float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0) > 0:
                metrics.setdefault("loss/rollout_field_floor", 0.0)
                metrics.setdefault("h3/rollout_field_ratio", 0.0)
            if float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0) > 0:
                metrics.setdefault("loss/rollout_null_anchor", 0.0)
        guidance_rescaled = False
        # A rollout-active step has already swapped its objective wholesale, and
        # nothing about a swapped objective is divided by a probability (the same
        # contract recipe mixing follows). Rescaling here would also rebuild the
        # loss from ``plain_result`` and silently discard the rollout terms.
        if use_guidance and args.h3_guidance_distillation_probability < 1.0 and not rollout_active:
            # The guidance objective replaces the ordinary one rather than adding
            # to it, so the unbiased sparse form keeps the ordinary loss every
            # step and scales only the guidance correction. Both terms reuse the
            # single trainable forward; no extra transformer pass is involved.
            plain_result = _velocity_loss(raw_prediction, inputs)
            loss = plain_result.loss + (result.loss - plain_result.loss) / args.h3_guidance_distillation_probability
            rescaled_velocity_loss = loss
            guidance_rescaled = True
        if args.h3_guidance_distillation_probability < 1.0:
            metrics["h3/guidance_distillation_active"] = float(use_guidance)
        base_preservation_term = None
        if reference_prediction is not None:
            preservation = joint_prediction_loss(
                raw_prediction,
                reference_prediction,
                video_mask=effective_video_mask,
                audio_mask=effective_audio_mask,
                video_sample_weight=video_sample_weight,
                audio_sample_weight=audio_sample_weight,
                balance=args.h3_loss_balance,
                mask_normalization=args.h3_loss_mask_normalization,
                video_weight=video_weight,
                audio_weight=audio_weight,
            )
            base_preservation_term = (
                args.h3_base_preservation_loss_weight / args.h3_base_preservation_probability
            ) * preservation.loss
            loss = loss + base_preservation_term
            metrics["loss/base_preservation"] = float(base_preservation_term.detach())
        if args.h3_base_preservation_loss_weight > 0:
            metrics["h3/base_preservation_active"] = float(preservation_active)
            metrics.setdefault("loss/base_preservation", 0.0)
        if null_anchor_student is not None and null_anchor_reference is not None:
            # A sparse anchor is an estimator of the dense one: the active term is
            # divided by its probability so the expected gradient is unchanged.
            null_anchor = (
                float(args.h3_guidance_null_anchor_weight)
                / null_anchor_probability
                * joint_prediction_loss(
                    null_anchor_student,
                    null_anchor_reference,
                    video_mask=effective_video_mask,
                    audio_mask=effective_audio_mask,
                    video_sample_weight=video_sample_weight,
                    audio_sample_weight=audio_sample_weight,
                    balance=args.h3_loss_balance,
                    mask_normalization=args.h3_loss_mask_normalization,
                    video_weight=video_weight,
                    audio_weight=audio_weight,
                ).loss
            )
            loss = loss + null_anchor
            metrics["loss/guidance_null_anchor"] = float(null_anchor.detach())
        elif float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0) > 0:
            # Present every step once the flag is on, so a run whose anchor never fired
            # is visible as a flat zero rather than as a missing tag nobody looks for.
            metrics["loss/guidance_null_anchor"] = 0.0
        if float(getattr(args, "h3_guidance_null_anchor_weight", 0.0) or 0.0) > 0 and null_anchor_probability < 1.0:
            metrics["h3/null_anchor_active"] = float(null_anchor_active)
        if use_crepa and self._crepa.active:
            crepa_loss, crepa_metrics = self._crepa.loss(
                batch.get("h3_dino_features"), update_similarity_threshold=crepa_update_similarity_threshold
            )
            loss = loss + crepa_loss
            metrics.update(crepa_metrics)
        elif use_crepa:
            metrics.update(self._crepa.status_metrics())
        if (
            base_preservation_term is not None
            or guidance_rescaled
            or dop_term is not None
            or rollout_replaced
            or null_anchor is not None
        ):
            average_loss = loss
            if base_preservation_term is not None:
                average_loss = average_loss - base_preservation_term
            if null_anchor is not None:
                # An auxiliary term like preservation, and removed for the same
                # reason: the reported average is the ordinary data loss a run
                # without the flag would have logged, so two runs stay comparable.
                average_loss = average_loss - null_anchor
            if guidance_rescaled:
                # Report the dense guidance objective so the running average stays
                # comparable across a sparse and a dense run.
                average_loss = average_loss - rescaled_velocity_loss + dense_loss
            if rollout_field_floor is not None:
                # An auxiliary term like the anchor: the reported average stays the
                # ordinary data loss a run without the flag would have logged.
                average_loss = average_loss - rollout_field_floor
            if rollout_null_anchor is not None:
                average_loss = average_loss - rollout_null_anchor
            if rollout_replaced:
                # The rollout SWAPS the objective on a drawn subset of steps
                # rather than estimating one objective sparsely, so nothing is
                # divided by the probability -- exactly as recipe mixing rescales
                # nothing. What would otherwise break is the running average,
                # which would jump between two objectives on different scales; it
                # is therefore reported as the ordinary data-velocity loss this
                # same forward already produced, which is what a run without the
                # flag would have logged.
                average_loss = average_loss - rescaled_rollout_loss + dense_loss
            metrics[LOSS_FOR_AVERAGE_KEY] = float(average_loss.detach())
        # Keep capture active until backward has completed. Non-reentrant
        # gradient checkpointing recomputes hooked blocks during backward and
        # requires the hook to perform the same tensor operations as forward.
        # begin_step() clears the captures before the next trainable pass.
        # Every consumer has run; nothing beyond this step may inherit the draw.
        self._step_mask = None
        self._step_row_video_timestep = None
        self._step_spatial_density_scale = None
        self._step_keyframes = None
        self._step_guides = None
        self._step_reference_modality = "av"
        self._step_qwen_control_dropout = False
        self._step_recipe = None
        return loss, metrics

    def call_dit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("MiniMax H3 uses its joint audio-video process_batch implementation")

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_h3_training_mode": args.h3_training_mode,
            "ss_h3_lora_token_refiner": str(args.h3_lora_token_refiner),
            "ss_h3_lora_targets": str(args.h3_lora_targets or "default"),
            "ss_h3_audio_only_spatial_tokens": str(args.h3_audio_only_spatial_tokens),
            "ss_h3_loss_balance": args.h3_loss_balance,
            "ss_h3_loss_mask_normalization": args.h3_loss_mask_normalization,
            "ss_h3_video_loss_weight": str(args.h3_video_loss_weight),
            "ss_h3_audio_loss_weight": str(args.h3_audio_loss_weight),
            "ss_h3_guide_specs": str(getattr(args, "h3_guide_specs", "") or "none"),
            "ss_h3_attn_auto_dispatch": str(args.h3_attn_auto_dispatch),
            "ss_h3_fused_indexed_adaln": str(args.h3_fused_indexed_adaln),
            "ss_h3_fused_swiglu": str(args.h3_fused_swiglu),
            "ss_h3_fused_elementwise": str(getattr(args, "h3_fused_elementwise", False)),
            "ss_h3_compile_attention": str(getattr(args, "h3_compile_attention", "inline")),
            "ss_h3_swiglu_chunk_rows": str(args.h3_swiglu_chunk_rows),
            "ss_h3_int8_attention": args.h3_int8_attention,
            "ss_h3_observed_modality": str(args.h3_observed_modality or "none"),
            "ss_h3_image_flow_shift": str(args.h3_image_flow_shift or "resolution_aware"),
            "ss_h3_guidance_distillation_scale": str(args.h3_guidance_distillation_scale or "one_pass"),
            "ss_h3_guidance_audio_scale": str(getattr(args, "h3_guidance_audio_scale", None) or "same"),
            "ss_h3_guidance_scale_range": (
                "fixed"
                if self._guidance_scale_range is None
                else f"{self._guidance_scale_range[0]},{self._guidance_scale_range[1]}"
            ),
            "ss_h3_guidance_distillation_probability": str(args.h3_guidance_distillation_probability),
            "ss_h3_overlay_weights": str(getattr(args, "h3_overlay_weights", None) or "none"),
            "ss_h3_overlay_weights_multiplier": str(getattr(args, "h3_overlay_weights_multiplier", 1.0)),
            "ss_h3_guidance_loss_form": args.h3_guidance_loss_form,
            "ss_h3_guidance_loss_schedule": args.h3_guidance_loss_schedule,
            "ss_h3_guidance_null_source": args.h3_guidance_null_source,
            "ss_h3_fuse_frozen_teachers": str(args.h3_fuse_frozen_teachers),
            "ss_h3_guidance_cfg_zero": str(args.h3_guidance_cfg_zero),
            "ss_h3_caption_dropout_rate": str(args.h3_caption_dropout_rate),
            "ss_h3_qwen_control_dropout_rate": str(args.h3_qwen_control_dropout_rate),
            "ss_h3_fp8_quantization_mode": args.h3_fp8_quantization_mode,
            "ss_h3_convrot_int8": str(args.h3_convrot_int8),
            "ss_h3_convrot_int8_bwd": args.h3_convrot_int8_bwd,
            "ss_h3_convrot_int8_fwd": args.h3_convrot_int8_fwd,
            "ss_h3_convrot_int8_lora_fused": str(args.h3_convrot_int8_lora_fused),
            "ss_h3_adaln_rank": str(args.h3_adaln_rank if args.h3_adaln_rank is not None else "full"),
            "ss_h3_reference_image_short_edge": str(args.reference_image_short_edge),
            "ss_h3_reference_image_size_mode": args.reference_image_size_mode,
            "ss_h3_reference_image_max_pixels": str(args.reference_image_max_pixels),
            "ss_h3_reference_video_short_edge": str(args.reference_video_short_edge),
            "ss_h3_reference_video_max_pixels": str(args.reference_video_max_pixels),
            "ss_h3_reference_video_fps": str(args.reference_video_fps),
            "ss_h3_text_visual_max_pixels": str(args.h3_text_visual_max_pixels),
            "ss_h3_max_caption_tokens": str(args.h3_max_caption_tokens),
            "ss_h3_extension_video_frames": str(args.h3_extension_video_frames),
            "ss_h3_extension_audio_latents": str(args.h3_extension_audio_latents),
            "ss_h3_extension_route": args.h3_extension_route,
            "ss_h3_extension_probability": str(args.h3_extension_probability),
            "ss_h3_mask_probability": str(args.h3_mask_probability),
            "ss_h3_frame_sigma_jitter": str(args.h3_frame_sigma_jitter),
            "ss_h3_spatial_density_jitter": str(args.h3_spatial_density_jitter),
            "ss_h3_keyframe_anchors": args.h3_keyframe_anchors or "none",
            "ss_h3_keyframe_random_count": str(args.h3_keyframe_random_count),
            "ss_h3_mask_mode": args.h3_mask_mode,
            "ss_h3_mask_audio": str(args.h3_mask_audio),
            "ss_h3_base_preservation_loss_weight": str(args.h3_base_preservation_loss_weight),
            "ss_h3_guidance_null_anchor_weight": str(getattr(args, "h3_guidance_null_anchor_weight", 0.0)),
            "ss_h3_guidance_null_anchor_probability": str(_anchor_probability(args)),
            "ss_h3_rollout_supervision": str(bool(getattr(args, "h3_rollout_supervision", False))),
            "ss_h3_rollout_probability": str(args.h3_rollout_probability),
            "ss_h3_rollout_steps": str(args.h3_rollout_steps),
            "ss_h3_rollout_window": str(args.h3_rollout_window),
            "ss_h3_rollout_stop_shifted": str(bool(getattr(args, "h3_rollout_stop_shifted", False))),
            "ss_h3_rollout_stop_min": str(float(getattr(args, "h3_rollout_stop_min", 0.0) or 0.0)),
            "ss_h3_adapter_prompt_only": str(bool(getattr(args, "h3_adapter_prompt_only", False))),
            "ss_h3_guidance_scale_sigma_max": str(float(getattr(args, "h3_guidance_scale_sigma_max", 1.0))),
            "ss_h3_adapter_ema_decay": str(float(getattr(args, "h3_adapter_ema_decay", 0.0) or 0.0)),
            "ss_h3_measured_variance_weighting": str(getattr(args, "h3_measured_variance_weighting", None) or "none"),
            "ss_h3_measured_variance_weight_max": str(float(getattr(args, "h3_measured_variance_weight_max", 4.0))),
            "ss_h3_rollout_teacher": str(getattr(args, "h3_rollout_teacher_config", None) or "none"),
            "ss_h3_rollout_teacher_privilege": str(getattr(args, "h3_rollout_teacher_privilege", "auto")),
            # Recorded although it defines no objective: it is the one rollout
            # knob that changes how the step is executed rather than what it
            # optimizes, and a reader comparing two runs' throughput needs it.
            "ss_h3_rollout_fused_teacher": str(bool(getattr(args, "h3_rollout_fused_teacher", False))),
            "ss_h3_rollout_field_floor": str(float(getattr(args, "h3_rollout_field_floor", 0.0) or 0.0)),
            "ss_h3_rollout_field_floor_direction": str(getattr(args, "h3_rollout_field_floor_direction", "self")),
            "ss_h3_rollout_field_cap": str(float(getattr(args, "h3_rollout_field_cap", 0.0) or 0.0)),
            "ss_h3_rollout_null_anchor_weight": str(float(getattr(args, "h3_rollout_null_anchor_weight", 0.0) or 0.0)),
            "ss_h3_rollout_prefix": str(getattr(args, "h3_rollout_prefix", "student")),
            "ss_h3_rollout_field_floor_sigma_max": str(float(getattr(args, "h3_rollout_field_floor_sigma_max", 1.0))),
            "ss_h3_base_preservation_probability": str(args.h3_base_preservation_probability),
            "ss_h3_dop_loss_weight": str(args.h3_dop_loss_weight),
            "ss_h3_dop_probability": str(args.h3_dop_probability),
            "ss_h3_dop_trigger": args.h3_dop_trigger or "none",
            "ss_h3_dop_class_prompt": args.h3_dop_class_prompt or "none",
            "ss_h3_shift_video": str(args.h3_shift_video),
            "ss_h3_shift_audio": str(args.h3_shift_audio),
            "ss_h3_sigma_sqrt_max_weight": str(args.h3_sigma_sqrt_max_weight),
            "ss_h3_timestep_sampling": args.timestep_sampling,
            "ss_h3_timestep_focus_min": str(args.h3_timestep_focus_min),
            "ss_h3_timestep_focus_max": str(args.h3_timestep_focus_max),
            "ss_h3_timestep_focus_probability": str(args.h3_timestep_focus_probability),
            "ss_h3_crepa": self._crepa_config.to_json() if self._crepa_config is not None else "disabled",
        }


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Train a MiniMax H3 LoRA with synchronized video and audio flow matching"
    parser.add_argument(
        "--h3_training_mode",
        choices=("fl2va", "ref2va", "ref2va_omni"),
        default="fl2va",
        help="select FL2VA, strict Ref2VA, or experimental zero-or-more-reference Ref2VA training",
    )
    parser.add_argument(
        "--h3_lora_token_refiner",
        action="store_true",
        help=(
            "also train LoRA adapters on the two H3 text token-refiner blocks; "
            "off by default and supported only by networks.lora_minimax_h3"
        ),
    )
    parser.add_argument(
        "--h3_lora_targets",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "select H3 LoRA groups and optional block ranges in one expression, for example "
            "'attention:0-13;mlp:3-5;audio'; omitted preserves the normal attention+MLP block target"
        ),
    )
    parser.add_argument(
        "--h3_audio_only_spatial_tokens",
        action="store_true",
        help=(
            "for audio-only target datasets, add one zero-initialized H3 spatial token per latent frame "
            "to the forward while keeping video loss disabled; opt-in because it changes the "
            "packed sequence relative to the native zero-video-row audio-only path"
        ),
    )
    parser.add_argument("--text_encoder", type=str, help="Qwen3-VL H3 BF16 checkpoint used only for sampling prompts")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=default_text_encoder_assets(),
        help="H3 tokenizer/processor directory used for sampling; defaults to the metadata bundled with Musubi",
    )
    parser.add_argument("--audio_vae", type=str, help="MiniMax H3 audio VAE checkpoint used only for sampling")
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4", "nvfp4", "nvfp4_awq"),
        default="none",
        help="optional Qwen3-VL quantization while pre-encoding sampling prompts",
    )
    parser.add_argument(
        "--h3_text_encoder_blocks_to_stream",
        type=int,
        default=0,
        help="stream this many of the 50 frozen Qwen3-VL layers from CPU while encoding sample prompts (CUDA only)",
    )
    parser.add_argument(
        "--h3_text_visual_max_pixels",
        type=int,
        default=0,
        help="maximum pixels per sampling control image presented to Qwen3-VL; 0 disables the cap",
    )
    parser.add_argument(
        "--h3_nvfp4_scaled_mm",
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use Blackwell W4A4 scaled_mm for a native NVFP4/AWQ sampling text encoder (PyTorch 2.10+)",
    )
    parser.add_argument(
        "--h3_loss_balance",
        choices=("token", "modality"),
        default="modality",
        help="combine joint AV loss over all valid latent elements or equally by modality means",
    )
    parser.add_argument(
        "--h3_loss_mask_normalization",
        choices=("weighted", "full"),
        default="weighted",
        help=(
            "soft-mask reduction: weighted divides by the sum of mask weights to keep loss scale stable; "
            "full divides by every latent element so lower coverage also lowers total gradient strength"
        ),
    )
    parser.add_argument("--h3_video_loss_weight", type=float, default=1.0)
    parser.add_argument("--h3_audio_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--h3_attn_auto_dispatch",
        action="store_true",
        help=(
            "prioritize cuDNN SDPA for large maskless CUDA BF16/FP16 attention shapes; "
            "short, masked, CPU, and FP32 workloads retain ordinary SDPA"
        ),
    )
    parser.add_argument(
        "--h3_observed_modality",
        type=str,
        default=None,
        choices=["video", "audio", "random"],
        help=(
            "Train one modality while the other is read as clean conditioning at the released "
            "transformer's own conditioning noise level. 'video' trains audio from video "
            "(video-to-audio / Foley); 'audio' trains video from audio; 'random' redraws the task "
            "each step across joint, video-observed and audio-observed, producing one adapter that "
            "keeps all three. Requires datasets that cache both modalities, and overrides the "
            "observed modality's loss weight to zero."
        ),
    )
    parser.add_argument(
        "--h3_image_flow_shift",
        type=float,
        default=None,
        help=(
            "override the default logit-normal, resolution-aware flow shift for image batches; "
            "video batches continue to use the released synchronized H3 schedule"
        ),
    )
    parser.add_argument(
        "--h3_guidance_distillation_scale",
        type=float,
        default=None,
        help="enable optional two-pass guidance-consistent training with an authoritative distillation scale",
    )
    parser.add_argument(
        "--h3_guidance_audio_scale",
        type=float,
        default=None,
        help=(
            "distillation scale for the AUDIO half of the guidance target; unset (default) uses the shared configured "
            "scale for both modalities, including a per-sample draw under --h3_guidance_scale_range. The audio field "
            "of the released checkpoints is not "
            "a clean amplification of a null-to-data difference (measured: its implied scale is not constant across "
            "sigma and its residual is nearly all systematic at the top), so an amplified audio target adds noise "
            "without a matching field to hold; 1.0 keeps the audio target plain while the video keeps the guidance "
            "form. Must be at least 1; ignored when the guidance objective is off"
        ),
    )
    parser.add_argument(
        "--h3_guidance_scale_range",
        type=str,
        default=None,
        metavar="LOWER,UPPER",
        help=(
            "draw the guidance-distillation scale uniformly in [LOWER, UPPER], once per micro-batch sample and step, "
            "instead of pinning the single --h3_guidance_distillation_scale (mutually exclusive with it). LOWER must "
            "be greater than 1 and no greater than UPPER. The draw uses its own distributed-synchronized generator, "
            "so adding the range leaves every other random branch of a seeded run untouched; validation reads the "
            "midpoint so its loss stays comparable. Composes with --h3_guidance_distillation_probability and with "
            "both --h3_guidance_loss_form and --h3_guidance_loss_schedule"
        ),
    )
    parser.add_argument(
        "--h3_overlay_weights",
        type=str,
        default=None,
        help=(
            "apply a LoRA (Musubi or Diffusers/PEFT keys) as a separate frozen module instead of merging it into the "
            "base weights; works on an INT8 ConvRot base, is excluded from the optimizer and saved checkpoints, and is "
            "active on all forwards including the base-preservation reference"
        ),
    )
    parser.add_argument(
        "--h3_overlay_weights_multiplier",
        type=float,
        default=1.0,
        help="strength of --h3_overlay_weights (default 1.0); requires --h3_overlay_weights",
    )
    parser.add_argument(
        "--h3_guidance_distillation_probability",
        type=float,
        default=1.0,
        help=(
            "probability of evaluating the empty-conditioning guidance branch on a batch; the guidance correction "
            "is divided by this probability to preserve the expected gradient, and the draw is synchronized across "
            "distributed ranks"
        ),
    )
    parser.add_argument(
        "--h3_extension_video_frames",
        type=int,
        default=0,
        help=(
            "leading latent video frames observed as context instead of generated, training video extension. "
            "The observed span is packed as clean condition rows and removed from the loss"
        ),
    )
    parser.add_argument(
        "--h3_extension_audio_latents",
        type=int,
        default=0,
        help="leading audio latents observed as context instead of generated, training audio extension",
    )
    parser.add_argument(
        "--h3_extension_probability",
        type=float,
        default=1.0,
        help=(
            "probability of training the extension recipe on a step; the remaining steps train the plain objective "
            "or, when masking is also configured, whichever recipe the shared per-step draw selects. Requires the "
            "extension flags and is synchronized across distributed ranks"
        ),
    )
    parser.add_argument(
        "--h3_frame_sigma_jitter",
        type=float,
        default=0.0,
        help=(
            "spread each latent frame's noise level around the step's shared schedule position by up to this much, "
            "so one forward supervises a range of the schedule instead of a single point"
        ),
    )
    parser.add_argument(
        "--h3_spatial_density_jitter",
        type=float,
        default=0.0,
        help=(
            "perturb the area normalization of the spatial RoPE grids by up to this fraction each step, drawn "
            "log-uniformly from [1/(1+j), 1+j], so fixed-resolution data still trains a range of token spacings. "
            "0 disables it"
        ),
    )
    parser.add_argument(
        "--h3_timestep_focus_min",
        type=float,
        default=0.4,
        help="lower edge of the unshifted base-sigma focus band",
    )
    parser.add_argument(
        "--h3_timestep_focus_max",
        type=float,
        default=0.8,
        help="upper edge of the unshifted base-sigma focus band",
    )
    parser.add_argument(
        "--h3_timestep_focus_probability",
        type=float,
        default=0.0,
        help=(
            "probability of sampling video/AV batches uniformly from the focus band instead of the full base-sigma "
            "range; image batches retain their resolution-aware schedule, and 0 preserves H3's default distribution"
        ),
    )
    parser.add_argument(
        "--h3_keyframe_anchors",
        type=str,
        default="",
        help=(
            "comma-separated conditioning frames given as clean context, each 'first', 'last', or a latent frame "
            "index, for example 'first,last' or '0,11,21'. Generalizes first/last keyframe conditioning to any set, "
            "training interpolation between arbitrary anchors"
        ),
    )
    parser.add_argument(
        "--h3_keyframe_random_count",
        type=int,
        default=0,
        help="draw this many distinct conditioning frames at random each step instead of listing them",
    )
    parser.add_argument(
        "--h3_guide_specs",
        type=str,
        default="",
        metavar="START:VIDEO_LATENTS:AUDIO_LATENTS[;...]",
        help=(
            "target-derived Ref2VA guide spans on the decoded pixel-frame timeline; for example '0:2:4;21:0:8' "
            "adds one AV guide and one audio-only guide. Negative START counts from the end. Visual starts must "
            "land on a cached VAE-window boundary and audio starts must be multiples of 3 pixel frames; zero selects "
            "no guide for that stream"
        ),
    )
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=(
            "short edge in pixels the Ref2VA reference caches were built with; it selects the matching reference "
            "cache keys and must equal the value given to latent caching"
        ),
    )
    parser.add_argument(
        "--reference_image_size_mode",
        choices=REFERENCE_IMAGE_SIZE_MODES,
        default="short_edge",
        help="Ref2VA image sizing used by both text and latent caches",
    )
    parser.add_argument(
        "--reference_image_max_pixels",
        type=int,
        default=0,
        help="optional target-area reference pixel cap; 0 uses the target bucket area",
    )
    parser.add_argument(
        "--reference_video_short_edge",
        type=int,
        default=REFERENCE_VIDEO_SHORT_EDGE,
        help="Ref2VA reference-video short edge used by both text and latent caches (default 768)",
    )
    parser.add_argument(
        "--reference_video_max_pixels",
        type=int,
        default=REFERENCE_VIDEO_MAX_PIXELS,
        help="maximum pixels per Ref2VA reference-video frame (default 768x1344)",
    )
    parser.add_argument(
        "--reference_video_fps",
        type=float,
        default=REFERENCE_VIDEO_FPS,
        help=(
            "reference-video subsampling rate in frames per source second the Ref2VA caches were built with; "
            "0 (default) selects the truncating caches and must equal the value given to both caching stages"
        ),
    )
    parser.add_argument(
        "--h3_mask_mode",
        choices=("off", "box", "border", "segment", "dataset"),
        default="off",
        help=(
            "video conditioning mask: box trains inpainting, border trains outpainting, segment hides a run of "
            "frames, all drawn procedurally per step; dataset instead reads the region the dataset authored via "
            "conditioning_mask_directory/conditioning_mask_path. The observed region is presented as clean context "
            "and excluded from the loss"
        ),
    )
    parser.add_argument(
        "--h3_mask_audio",
        action="store_true",
        help="also hide a contiguous run of audio latents, training audio inpainting alongside the video mask",
    )
    parser.add_argument(
        "--h3_mask_probability",
        type=float,
        default=1.0,
        help=(
            "probability of training the masked recipe on a step; the remaining steps train the plain objective "
            "or, when extension is also configured, whichever recipe the shared per-step draw selects. Requires "
            "--h3_mask_mode or --h3_mask_audio and is synchronized across distributed ranks"
        ),
    )
    parser.add_argument(
        "--h3_mask_min_fraction",
        type=float,
        default=0.25,
        help="smallest fraction of each masked axis the generated region may cover",
    )
    parser.add_argument(
        "--h3_mask_max_fraction",
        type=float,
        default=0.75,
        help="largest fraction of each masked axis the generated region may cover",
    )
    parser.add_argument(
        "--h3_extension_route",
        choices=("condition_rows", "per_row_sigma"),
        default="condition_rows",
        help=(
            "how the observed context is presented. condition_rows duplicates it as clean rows, generalizing the "
            "released keyframe contract. per_row_sigma pins the observed rows inside the target block, costing no "
            "extra tokens but placing intra-block noise levels outside what the released weights have seen"
        ),
    )
    parser.add_argument(
        "--h3_max_caption_tokens",
        type=int,
        default=0,
        help="caption-token cap used to build the H3 text cache; 0 expects uncapped caches",
    )
    parser.add_argument(
        "--h3_caption_dropout_rate",
        type=float,
        default=0.0,
        help=(
            "probability of replacing the prompt with the cached empty conditioning for a step, training the "
            "unconditional branch; requires --cache_guidance_empty. Steps that drop the caption skip the "
            "guidance-consistent correction, which has nothing to invert without a prompt"
        ),
    )
    parser.add_argument(
        "--h3_qwen_control_dropout_rate",
        type=float,
        default=0.0,
        help=(
            "EXPERIMENTAL probability of presenting a step without its qwen_control_* visuals, CFG style; requires a "
            "text cache written with --h3_qwen_control_dropout. 0 (default) always shows the controls"
        ),
    )
    parser.add_argument(
        "--h3_guidance_loss_form",
        choices=("normalized", "contrastive"),
        default="normalized",
        help=(
            "normalized applies flow loss to the reconstructed conditional field; contrastive applies the equivalent "
            "scale-squared loss magnitude of a direct extrapolated target"
        ),
    )
    parser.add_argument(
        "--h3_guidance_loss_schedule",
        choices=("sigma", "constant"),
        default="sigma",
        help=(
            "sigma uses effective_scale = 1 + (configured_scale - 1) * modality_sigma; "
            "constant applies the configured guidance scale at every noise level"
        ),
    )
    parser.add_argument(
        "--h3_guidance_null_source",
        choices=("live", "frozen"),
        default="live",
        help=(
            "live evaluates the null-conditioning branch with the trainable adapter active; frozen disables the "
            "adapter for that forward so the guidance correction inverts a fixed base field"
        ),
    )
    parser.add_argument(
        "--h3_guidance_cfg_zero",
        action="store_true",
        help=(
            "CFG-Zero* style rescale of the null branch before the guidance form is applied: each sample and "
            "modality projects the null prediction onto the conditional field"
        ),
    )
    parser.add_argument(
        "--h3_fuse_frozen_teachers",
        action="store_true",
        help=(
            "EXPERIMENTAL batch the frozen empty-guidance and base-preservation teachers into one transformer forward; "
            "requires frozen guidance and active base preservation, and may use more peak VRAM"
        ),
    )
    parser.add_argument(
        "--h3_validation_field_probe",
        action="store_true",
        help=(
            "DEBUG. Reported against the frozen base checkpoint, so it is incompatible with --base_weights, which "
            "merges an adapter into that checkpoint. val/velocity_err is an ENERGY ratio (squared error over squared "
            "target), so two runs are comparable at equal values but the axis is not linear -- interpolate on its "
            "square root, not on it. "
            "Report what the adapter has done to the guidance field, measured against the frozen base "
            "on the validation items. Diagnostic only, subject to removal, and not part of any recommended "
            "recipe. Requires a validation set and empty-text caches (--cache_guidance_empty)"
        ),
    )
    parser.add_argument(
        "--h3_validation_rollout_probe",
        type=int,
        default=0,
        help=(
            "roll the adapter and the frozen base out from the same noise for this many Euler steps and report, at "
            "the states reached, how wrong the adapter's own clean-clip estimate is and how much of the guidance "
            "field survives THERE. The field probe beside it measures on noised data states, which generation never "
            "visits, so a method that works by correcting where the sampler goes cannot register in it. Costs "
            "2*steps+4 no-grad forwards per validation, paid once per dataset rather than per item and bin. 0 "
            "disables"
        ),
    )
    parser.add_argument(
        "--h3_validation_rollout_stop",
        type=float,
        default=0.5,
        help=(
            "unshifted sigma the validation rollout walks down to (default 0.5); lower goes further along the "
            "trajectory and costs the same, higher stays nearer the noise the walk started from"
        ),
    )
    parser.add_argument(
        "--h3_profile_steps",
        type=int,
        default=0,
        help=(
            "profile this many optimizer steps with torch.profiler once three warm-up steps have passed and report "
            "one table of device time per kernel category (attention, GEMM, host<->device swap, elementwise/other) "
            "plus the idle time outside the union of all kernel intervals, to the log and to "
            "<output_dir>/<output_name>_profile.txt. Category shares can add up to more than the busy time because "
            "streams overlap. The run must be at least 3 + N optimizer steps long. Profiling stops afterwards and "
            "training continues. Diagnostic only. 0 disables"
        ),
    )
    parser.add_argument(
        "--h3_rollout_supervision",
        action="store_true",
        help=(
            "EXPERIMENTAL truncated on-policy rollout supervision (D-OPSD). On a drawn subset of steps the VIDEO "
            "objective is replaced: the model's own sampler runs --h3_rollout_steps no-grad Euler steps from pure "
            "noise to a randomly drawn stopping sigma, and the next --h3_rollout_window states are supervised "
            "against the frozen base conditioned on the target clip's own frames (--h3_rollout_teacher_config). "
            "Audio keeps its ordinary data loss on those steps, because the Qwen conditioner has no audio path and "
            "the teacher was measured to carry no audio advantage"
        ),
    )
    parser.add_argument(
        "--h3_rollout_teacher_config",
        type=str,
        default=None,
        help=(
            "second dataset TOML whose text-encoder cache carries the privileged teacher conditioning -- the SAME "
            "items as --dataset_config, privileged either by the target clip's frames attached as qwen_control_* "
            "assets (variant A) or by extra target frames cached as additional references (variant B). Target "
            "latents always come from the training batch. Pairing is by item key and an unpaired or unprivileged "
            "corpus fails before the model loads"
        ),
    )
    parser.add_argument(
        "--h3_rollout_teacher_privilege",
        type=str,
        choices=TEACHER_PRIVILEGE_CHANNELS,
        default="auto",
        help=(
            "which channel carries the teacher's advantage. auto reads the caches: qwen_control_* visuals in the "
            "teacher's TEXT cache (variant A), otherwise strictly more reference entries in its LATENT cache than "
            "the student holds for the same item (variant B). Pin the channel only to measure one of two that a "
            "corpus happens to carry both of"
        ),
    )
    parser.add_argument(
        "--h3_rollout_probability",
        type=float,
        default=_ROLLOUT_PROBABILITY_DEFAULT,
        help=(
            "probability that a step trains the rollout objective instead of the data one for video. The draw is "
            "synchronized across distributed ranks. No loss is divided by it: the objective is swapped, not "
            "sparsely estimated, so the reported average loss is the ordinary data-velocity loss instead"
        ),
    )
    parser.add_argument(
        "--h3_rollout_steps",
        type=int,
        default=_ROLLOUT_STEPS_DEFAULT,
        help=(
            "no-grad Euler steps of the model's own sampler, from pure noise to the drawn stopping sigma, that "
            "carry the state on-policy before supervision starts. Each costs one forward and no memory beyond the "
            "state itself"
        ),
    )
    parser.add_argument(
        "--h3_rollout_window",
        type=int,
        default=_ROLLOUT_WINDOW_DEFAULT,
        help=(
            f"supervised sub-steps taken after the on-policy state, at most {MAX_ROLLOUT_WINDOW}. Each costs one "
            "trainable forward and one no-grad teacher forward, and the state advances between them under "
            "stop-grad from the student's own prediction, so a long window drifts away from the states the "
            "teacher's advantage was measured on"
        ),
    )
    parser.add_argument(
        "--h3_rollout_stop_shifted",
        action="store_true",
        help=(
            "draw the rollout's stopping point uniformly over the SHIFTED video sigma range instead of the "
            "unshifted base grid, then invert the shift to reach it. With --h3_shift_video 12 a uniform base draw "
            "lands almost every stop above shifted sigma 0.9, so the mid band the D-OPSD preflight measured the "
            "teacher to be strongest in (-50% relative error at .6) is never supervised. Reaching the lower base "
            "sigmas this produces needs more --h3_rollout_steps for the state to stay on-policy"
        ),
    )
    parser.add_argument(
        "--h3_rollout_fused_teacher",
        action="store_true",
        help=(
            "run each supervised sub-step's student and teacher forwards through ONE pass over the transformer "
            "blocks instead of two. The two arms pack different sequences -- the teacher's presentation is the "
            "longer one -- so they are interleaved block by block rather than batched, which leaves every arm's "
            "attention exactly what it was and makes --blocks_to_swap stream each swapped block once per sub-step "
            "instead of twice. Off by default so the two paths stay A/B comparable"
        ),
    )
    parser.add_argument(
        "--h3_rollout_field_floor",
        type=float,
        default=0.0,
        help=(
            "weight of a LENGTH floor on the guidance field at each supervised rollout state: with the frozen "
            "base's empty-prompt prediction e and prompted prediction g evaluated at the same state, the student's "
            "prompted prediction g' is penalised by weight * relu(1 - ||g' - e|| / ||g - e||)^2, so the "
            "prompted-to-empty gap may not shrink below the checkpoint's but is free to grow and free to turn. The "
            "direction of the field belongs to the concept being learned and is left to the teacher term; only its "
            "length, which belongs to the distillation, is held. The null branch in both fields is the FROZEN one, "
            "so the floor cannot be satisfied by moving the student's own empty branch -- and, by the same token, it "
            "does not bound the gap to that branch: pair it with --h3_guidance_null_anchor_weight, which holds the "
            "student's empty branch at the frozen one, or the floor is only a displacement floor on the prompted "
            "prediction. Costs two no-grad frozen "
            "forwards per supervised sub-step (folded into the fused pass under --h3_rollout_fused_teacher). "
            "Requires --h3_rollout_supervision and a text cache built with --cache_guidance_empty. 0 disables"
        ),
    )
    parser.add_argument(
        "--h3_guidance_scale_sigma_max",
        type=float,
        default=1.0,
        help=(
            "cap on the guidance target's schedule: at a shifted sigma above this value the distillation scale is 1 "
            "and the step trains on the plain data target. The released checkpoints' implied scale runs to 11-16 "
            "above sigma 0.9 where the data are nearly noise. 1.0 (default) applies the schedule everywhere"
        ),
    )
    parser.add_argument(
        "--h3_validation_multipliers",
        type=str,
        default="",
        help=(
            "comma-separated LoRA multipliers (e.g. 0.75,1.25) at which validation repeats its data-state pass and "
            "reports val/m<multiplier>/velocity_err_rel, field, field_cos and drift/prompted_rel: how the adapter "
            "behaves when its strength is turned at inference, the analogue of a guidance-scale sensitivity curve. "
            "The rollout probe is not repeated. Each multiplier costs one more validation pass"
        ),
    )
    parser.add_argument(
        "--h3_adapter_ema_decay",
        type=float,
        default=0.0,
        help=(
            "keep an exponential moving average of the adapter's trainable parameters with this decay, updated after "
            "every optimizer step, and save it beside each scheduled checkpoint as <output_name>-ema-step<N>.safetensors "
            "(an ordinary adapter file). Damps the step-to-step swing between competing loss terms without changing "
            "what is optimised. 0 (default) keeps no average"
        ),
    )
    parser.add_argument(
        "--h3_validate_ema",
        action="store_true",
        help="run validation on the EMA adapter (--h3_adapter_ema_decay) instead of the live one",
    )
    parser.add_argument(
        "--h3_adapter_prompt_only",
        action="store_true",
        help=(
            "run every EMPTY-prompt forward with the adapter switched off, so the student's empty branch is the "
            "frozen checkpoint's by construction. A guidance-distilled checkpoint never evaluates its empty branch at "
            "inference, so nothing is lost there; in training the live-null degeneracy (empty branch chasing the "
            "clip) becomes impossible and --h3_guidance_null_anchor_weight has nothing left to hold, which saves its "
            "two forwards a step. Rejected with the anchor and with --h3_caption_dropout_rate above 0, whose "
            "dropped steps would train nothing"
        ),
    )
    parser.add_argument(
        "--h3_measured_variance_weighting",
        type=str,
        default=None,
        help=(
            "path to a probe_g_curve JSON report; weight every sample's loss by the inverse of the target DISPERSION "
            "PROXY measured at its shifted sigma (raw^2 - g^2 per bucket, per modality; a proxy, not a variance), "
            "normalised to mean 1 over the buckets and capped by --h3_measured_variance_weight_max. A heuristic "
            "sigma reweighting that counts noisy noise levels for less and so changes the objective; composes "
            "multiplicatively with --weighting_scheme"
        ),
    )
    parser.add_argument(
        "--h3_measured_variance_weight_max",
        type=float,
        default=4.0,
        help="cap on a measured inverse-variance weight after normalisation to mean 1 (default 4.0)",
    )
    parser.add_argument(
        "--h3_rollout_stop_min",
        type=float,
        default=0.0,
        help=(
            "lower bound of the rollout stop draw, on the coordinate the draw is made on (unshifted base, or the "
            "shifted video sigma under --h3_rollout_stop_shifted). Without --h3_rollout_stop_shifted both this and "
            "--h3_timestep_focus_max are unshifted base sigmas, and focus_max <= stop_min gives the data term and the "
            "rollout term disjoint noise bands; with it the two live on different coordinates and must be converted "
            "before any such claim. 0 (default) draws the whole range"
        ),
    )
    parser.add_argument(
        "--h3_rollout_prefix",
        choices=("student", "teacher"),
        default="student",
        help=(
            "who walks the no-gradient prefix of the rollout (the --h3_rollout_steps Euler steps from noise) before "
            "the supervised window. 'student' (default) is on-policy: the adapter walks and the teacher is asked at "
            "the states it reaches. 'teacher' walks the prefix with the FROZEN privileged teacher instead, so the "
            "supervised window starts from states the teacher's own trajectory produced and the student refines "
            "them for --h3_rollout_window sub-steps (a hybrid policy: a window of 1 is off-policy anchoring, a "
            "longer window approaches on-policy). Asking the teacher at states the student made from a conditioning "
            "it does not share (a clean keyframe beside content the student invented) hands it a conflicted input; "
            "the teacher prefix avoids that. Same forward count"
        ),
    )
    parser.add_argument(
        "--h3_rollout_null_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "the null anchor at the supervised ROLLOUT states: at each one the student's empty-prompt prediction "
            "(adapter on, gradient) is held to the frozen base's empty prediction there, weighted by this value. The "
            "data-step anchor (--h3_guidance_null_anchor_weight) costs two forwards on every prompted step; this one "
            "costs one graded forward per supervised state on rollout steps only, and the frozen empty arm it needs "
            "is the one --h3_rollout_field_floor already evaluates. Use it with the data-step anchor at 0 to move "
            "the anchor onto the states generation walks, or with both for belt and braces. Requires "
            "--h3_rollout_supervision and --cache_guidance_empty; rejected with --h3_adapter_prompt_only. 0 disables"
        ),
    )
    parser.add_argument(
        "--h3_rollout_field_cap",
        type=float,
        default=0.0,
        help=(
            "upper bound on the field-length ratio the floor scores, as a multiple of the checkpoint's field length; "
            "adds relu(ratio - cap)^2 under the floor's weight so the field is held in a band [1, cap] rather than "
            "only from below. An over-long field shows as burnt colour and contrast, and a length that has been "
            "co-adapted into the weights is not undone by an inference multiplier. Under "
            "--h3_rollout_field_floor_direction teacher the ratio is the signed projection onto the teacher's direction, "
            "and the cap bounds that projection rather than the field's total length. Requires --h3_rollout_field_floor "
            "above 0 and a value above 1. 0 (default) disables the cap"
        ),
    )
    parser.add_argument(
        "--h3_rollout_field_floor_direction",
        choices=("self", "teacher"),
        default="self",
        help=(
            "which direction the field floor measures the student's field along. 'self' (default) floors the plain "
            "length ||g' - e||, whose gradient lengthens the field along the student's OWN current direction -- "
            "which at a sampler state has usually drifted, so the floor lengthens the error. 'teacher' floors the "
            "PROJECTION of the student's field onto the direction of the privileged teacher's field at the same "
            "state, <g' - e, t> / ||t|| against ||g - e|| with t = teacher - e: lengthening along a wrong direction "
            "earns nothing, and the gradient pulls toward the teacher's direction, so the floor lengthens and turns "
            "the field at once. No extra forward: the teacher is already evaluated at that state"
        ),
    )
    parser.add_argument(
        "--h3_rollout_field_floor_sigma_max",
        type=float,
        default=1.0,
        help=(
            "apply the field floor only at supervised states whose SHIFTED video sigma is at most this value; "
            "sub-steps above it are left out of the floor's mean. The checkpoint's implied guidance scale runs to "
            "11-16 above sigma 0.9 where the data are nearly noise, and a floor there pushes into noise. 1.0 (default) "
            "applies it everywhere"
        ),
    )
    parser.add_argument(
        "--h3_guidance_null_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "DEBUG. Penalise movement of the EMPTY-prompt prediction away from the base checkpoint's, weighted by "
            "this value, while leaving the prompted prediction free. Unlike "
            "--h3_base_preservation_loss_weight it does not constrain the prompted branch, so it does not fight the "
            "data term directly; it constrains a degree of freedom that learning a concept does not need. Measured "
            "on one corpus pair, at matched amounts of learning, it left the PROMPTED prediction about 0.03 closer "
            "to the checkpoint's on val/drift/prompted_rel, and won a six-pair blind render comparison 6-0 against "
            "ordinary training. Read val/drift/prompted_rel to judge it: val/field and its relatives are built on "
            "the prompted-minus-empty difference, which this term directly holds, so they flatter it. Costs one "
            "extra grad forward and one no-grad forward, about 1.8x a step, and reaches a given fit in more steps. "
            "Skipped on caption-dropout steps. One seed, small corpora; treat as experimental. 0 disables"
        ),
    )
    parser.add_argument(
        "--h3_guidance_null_anchor_probability",
        type=float,
        default=1.0,
        help=(
            "evaluate the null anchor on this synchronized random fraction of prompted steps and divide the active "
            "term by it, so the expected gradient is the dense anchor's at a fraction of its two forwards a step "
            "(1.8x -> 1 + 0.8p). Its own random stream, like the other sparse branches. 1 (default) anchors every step"
        ),
    )
    parser.add_argument(
        "--h3_base_preservation_loss_weight",
        type=float,
        default=0.0,
        help=(
            "optional frozen-base prediction-preservation loss weight; adds one no-grad transformer forward on each "
            "batch selected by --h3_base_preservation_probability"
        ),
    )
    parser.add_argument(
        "--h3_base_preservation_probability",
        type=float,
        default=1.0,
        help=(
            "probability of evaluating the frozen-base preservation branch on a batch; active losses are divided by "
            "this probability to preserve the expected gradient, and the draw is synchronized across distributed ranks"
        ),
    )
    parser.add_argument(
        "--h3_dop_loss_weight",
        type=float,
        default=0.0,
        help="optional Differential Output Preservation weight under trigger-free cached conditioning",
    )
    parser.add_argument(
        "--h3_dop_probability",
        type=float,
        default=1.0,
        help="probability of a DOP step; active loss is divided by this value",
    )
    parser.add_argument("--h3_dop_trigger", type=str, default="", help="trigger used to identify the matching DOP cache")
    parser.add_argument("--h3_dop_class_prompt", type=str, default="", help="class phrase used to identify the matching DOP cache")
    parser.add_argument(
        "--crepa",
        nargs="*",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "enable temporal representation alignment; optional values: student_block=16 teacher_block=33 "
            "weight=0.05 tau=1 neighbors=2 schedule=constant warmup_steps=0 max_steps=0 normalize=true "
            "cutoff_step=0 similarity_ema_decay=0.99 threshold_mode=permanent"
        ),
    )
    parser.add_argument(
        "--int8_convrot_base",
        action="store_true",
        help="load the pruned Comfy INT8 ConvRot transformer for LoRA training",
    )
    parser.add_argument(
        "--h3_convrot_int8",
        action="store_true",
        help=(
            "quantize the released BF16 transformer to ConvRot INT8 as it loads, rather than reading a checkpoint "
            "that was quantized offline. Because the quantization happens after the weight transforms, it composes "
            "with --h3_adaln_rank and so quantizes a reduced AdaLN instead of the full-width projections the "
            "published pruned checkpoints carry"
        ),
    )
    parser.add_argument(
        "--h3_convrot_int8_bwd",
        choices=("bf16", "int8"),
        default="bf16",
        help="precision of the ConvRot backward pass; int8 is faster and coarser",
    )
    parser.add_argument(
        "--h3_convrot_int8_fwd",
        choices=("int8", "bf16"),
        default="int8",
        help=(
            "how the ConvRot forward evaluates its matmul. 'int8' rotates the activations and runs the fused "
            "INT8 kernel; 'bf16' undoes the rotation on the weight instead and hands the vendor GEMM an ordinary "
            "matrix. The stored weights and the arithmetic result are the same either way, so this trades "
            "quantized compute for a better-tuned kernel and is worth measuring on GPUs with fast BF16"
        ),
    )
    parser.add_argument(
        "--h3_convrot_int8_lora_fused",
        action="store_true",
        help=(
            "fuse the LoRA-up projection into the ConvRot INT8 dequantization epilogue; requires online or "
            "pre-quantized ConvRot INT8 weights with INT8 forward and backward, and automatically falls back for "
            "LoRA dropout or split dimensions"
        ),
    )
    parser.add_argument(
        "--h3_fp8_quantization_mode",
        choices=("block", "channel", "tensor"),
        default="block",
        help=(
            "granularity of the scale that accompanies each FP8 weight. Block is the finest and the default; the "
            "measured difference between them is small because FP8 error is dominated by the mantissa rather than "
            "the scale"
        ),
    )
    parser.add_argument(
        "--h3_adaln_rank",
        type=int,
        default=None,
        help=(
            "reduce the AdaLN timestep projection to this rank while loading, shrinking the frozen base by ~13B "
            "parameters; the reduced weights stay in BF16 because they are no longer large enough to be worth quantizing"
        ),
    )
    parser.add_argument(
        "--h3_fused_qk_norm_rope",
        action="store_true",
        help=(
            "use the opt-in Triton kernel that fuses H3 per-head Q/K RMSNorm with split RoPE; "
            "unsupported shapes and torch.compile automatically use the eager/Inductor path"
        ),
    )
    parser.add_argument(
        "--h3_fused_indexed_adaln",
        action="store_true",
        help=(
            "use an opt-in Triton kernel that fuses each main-block RMSNorm with token-indexed AdaLN shift/scale; "
            "frozen LoRA bases use the fused forward/backward path and unsupported cases fall back safely"
        ),
    )
    parser.add_argument(
        "--h3_fused_swiglu",
        action="store_true",
        help=(
            "use an opt-in Triton kernel for the SwiGLU activation in H3 main and token-refiner feed-forward layers; "
            "unsupported cases fall back safely and compiled blocks use their Inductor path"
        ),
    )
    parser.add_argument(
        "--h3_fused_elementwise",
        action="store_true",
        help=(
            "apply the AdaLN modulation, the gated residual adds and a LoRA delta add whose factor is not a power of two "
            "as single addcmul / add(alpha=) kernels: the product stays in the fp32 accumulator and is rounded once, so "
            "results differ from the default two-kernel form by bf16 rounding (typically the last bit); saves about a "
            "third of the elementwise memory traffic of a block in eager mode"
        ),
    )
    parser.add_argument(
        "--h3_compile_attention",
        choices=("inline", "auto", "opaque"),
        default="inline",
        help=(
            "with --compile, how the attention kernel call is presented to Dynamo. 'inline' (default) traces it, which "
            "is the partitioning every existing --compile run has; 'opaque' hides it behind torch.compiler.disable so a "
            "block compiles to exactly one graph before and one after the kernel (for FlashAttention bindings Dynamo "
            "cannot trace, whose internal graph break otherwise cascades into unfusable fragments); 'auto' is opaque for "
            "--flash_attn/--flash3 and inline for --sdpa and under --compile_fullgraph, which forbids any break. "
            "Opt in after checking the compile log for graph breaks inside the attention call"
        ),
    )
    parser.add_argument(
        "--h3_swiglu_chunk_rows",
        type=int,
        default=0,
        help=(
            "split each H3 main-block feed-forward operation into at most this many sequence rows to reduce peak VRAM; "
            "0 disables chunking"
        ),
    )
    parser.add_argument(
        "--h3_gradient_checkpointing_blocks",
        type=int,
        default=None,
        help=(
            "checkpoint only the last N of H3's 50 main blocks; default checkpoints all blocks. "
            "Lower values trade more VRAM for less recomputation and require resident eager blocks"
        ),
    )
    parser.add_argument(
        "--h3_checkpoint_keep",
        choices=("none", "attention", "qkv"),
        default="none",
        help=(
            "what block gradient checkpointing keeps instead of recomputing: 'attention' keeps each block's fused "
            "attention output so the recompute skips the attention forward; 'qkv' also keeps the base QKV projection "
            "output. Costs one (or four) rows-by-hidden activations per checkpointed block and needs a fused attention "
            "kernel (SDPA flash/cuDNN/efficient or registered flash-attn ops); a batch that falls back to SDPA's math "
            "backend stops with an error. Requires --gradient_checkpointing; incompatible with "
            "--gradient_checkpointing_cpu_offload, --compile, --h3_int8_attention train, block-sparse attention, "
            "--block_swap_granularity layer and, for 'qkv', --h3_convrot_int8_lora_fused"
        ),
    )
    parser.add_argument(
        "--h3_gradient_checkpointing_cpu_offload_pin_memory",
        action="store_true",
        help=(
            "pin H3 CPU-offloaded checkpoint activations for faster transfers; requires substantial non-pageable host RAM "
            "and --gradient_checkpointing --gradient_checkpointing_cpu_offload"
        ),
    )
    parser.add_argument(
        "--h3_reusable_activation_offload",
        action="store_true",
        help=(
            "reuse pinned CPU buffers for checkpoint activations and prefetch them in reverse block order; "
            "requires --gradient_checkpointing --gradient_checkpointing_cpu_offload"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing_cpu_offload_dtype",
        choices=("none", "fp8_e4m3"),
        default="none",
        help=(
            "wire dtype for CPU-offloaded checkpoint activations; 'fp8_e4m3' halves PCIe traffic and pinned host memory "
            "for bf16/fp16 activations at the cost of a slightly lossy recomputation. Requires "
            "--gradient_checkpointing --gradient_checkpointing_cpu_offload --h3_reusable_activation_offload"
        ),
    )
    parser.add_argument(
        "--h3_block_sparse_kv_fraction",
        type=float,
        default=0.0,
        help=(
            "share of key blocks each query block attends to; 0 disables block-sparse attention "
            "and 1.0 keeps every block, which reproduces dense attention"
        ),
    )
    parser.add_argument(
        "--h3_block_sparse_threshold",
        type=float,
        default=0.0,
        help=(
            "keep the highest scoring key blocks until they hold this share of the score mass, "
            "instead of a fixed count; takes precedence over --h3_block_sparse_kv_fraction"
        ),
    )
    parser.add_argument(
        "--h3_block_sparse_start_block",
        type=int,
        default=0,
        help="index of the first transformer block to run block-sparse; earlier blocks stay dense",
    )
    parser.add_argument(
        "--h3_block_sparse_block_shape",
        type=str,
        default=None,
        help=(
            "lattice tile for block-sparse selection as frames,height,width whose product is the block size "
            "(128), e.g. 1,8,16; text, audio, and reference rows stay dense"
        ),
    )
    parser.add_argument(
        "--h3_int8_attention",
        choices=("off", "aux", "train"),
        default="off",
        help=(
            "experimental H3-owned INT8 attention: 'aux' applies it only to guidance/base-preservation teacher "
            "forwards, while 'train' also uses its optimized backward for the trainable forward; default 'off' "
            "leaves the selected SDPA/FlashAttention backend unchanged"
        ),
    )
    parser.add_argument(
        "--h3_shift_video",
        type=float,
        default=VIDEO_FLOW_SHIFT,
        help="exponential flow shift for the target video stream (H3 released schedule: 12.0)",
    )
    parser.add_argument(
        "--h3_shift_audio",
        type=float,
        default=AUDIO_FLOW_SHIFT,
        help="exponential flow shift for the target audio stream (H3 released schedule: 3.0)",
    )
    parser.add_argument(
        "--h3_sigma_sqrt_max_weight",
        type=float,
        default=10.0,
        help="maximum inverse-square loss weight used by --weighting_scheme sigma_sqrt (default: 10.0)",
    )
    parser.set_defaults(
        network_module="networks.lora_minimax_h3",
        mixed_precision="bf16",
        # --timestep_sampling selects only the shape of the unshifted
        # coordinate; H3's per-modality shifts are applied on top of it. A
        # uniform base keeps usable sampling density at low sigma once the
        # video shift is applied.
        timestep_sampling="uniform",
        discrete_flow_shift=1.0,
        vae_dtype="float32",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    args = read_config_from_file(args, parser)
    args.dit_dtype = None
    trainer = MiniMaxH3NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
