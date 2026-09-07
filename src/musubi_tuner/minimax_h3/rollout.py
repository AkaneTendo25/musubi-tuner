"""Truncated on-policy rollout supervision (D-OPSD) for MiniMax H3.

The ordinary flow objective supervises the model at states built from *data*:
a clean latent is mixed with noise at one sampled sigma, and the target is the
data velocity ``x0 - noise``. Those states are on the data manifold by
construction, and the model never sees the states its own sampler actually
visits. A short rollout closes that gap: run the student's own sampler from pure
noise for a few no-grad Euler steps, then supervise it *there* -- on-policy --
against a teacher evaluated at the same state.

The teacher here is privileged rather than larger. H3's Qwen3-VL conditioner
accepts in-context visuals, so a text cache built with frames of the target clip
attached as ``qwen_control_*`` assets conditions the *same frozen weights* on
information the student does not have. ``experiments/nullfield/dopsd_preflight.py``
measured that advantage on the frozen base: 22-50% lower relative velocity error
on video, and none at all on audio, because Qwen has no audio path. That
measurement is the reason this module supervises video only and leaves audio on
its ordinary data loss.

The privilege does not have to travel through Qwen. Ref2VA conditions the DiT on
encoded *reference* rows, so a teacher dataset cached with extra target frames
declared as references sees more of the clip than the student does through a
channel Qwen never touches -- the same asymmetry, moved from the text cache into
the latent cache. Both channels are supported and told apart by what the caches
actually contain, because that is the only thing that decides whether a teacher
is privileged at all.

Everything here is deliberately free of the trainer: the schedule is arithmetic
on sigmas, the Euler step is one line of the velocity convention, and the
teacher cache is a read-only view of a second dataset's cached outputs.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from safetensors import safe_open

from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_KEYFRAME_VISUALS_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    logical_cache_key,
)

logger = logging.getLogger(__name__)

# The window token cache file names carry: frame position, frame count and an
# optional section index ("00000-073", "00000-073-01"). Rollout item keys end in
# one whenever the dataset cut windows out of a source, and stripping it gives
# the source's own key back.
_WINDOW_SUFFIX = re.compile(r"_\d{5}-\d{3}(?:-\d+)?$")

# Batch key carrying one item key per batch member. BucketBatchManager emits it
# only when a consumer asks for it by name, so no other architecture's batch
# schema changes.
H3_ROLLOUT_ITEM_KEYS_BATCH_KEY = "mmh3_item_keys"

# The text-side tensors a teacher forward substitutes. The prompt branch only:
# the ROLLOUT never evaluates an empty teacher, and on an active step the
# student's own empty presentation must stay the student's.
TEACHER_REQUIRED_KEYS = (H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY)
# Presentation markers that travel with the hidden states. The control marker is
# what makes the teacher a teacher; the keyframe marker is carried so a corpus
# cached with both presentations is not silently stripped of one. The
# conditioning task rides along because an endpoint teacher differs from its
# student in exactly that scalar: same clip, same latents, same everything, but
# the teacher's cache declares ``fl2va`` where the student's declares ``t2va``,
# and that is what turns the keyframe rows already present in the shared latent
# cache into conditioning the teacher reads and the student does not.
TEACHER_OPTIONAL_KEYS = (H3_QWEN_CONTROL_VISUALS_KEY, H3_KEYFRAME_VISUALS_KEY, H3_CONDITIONING_TASK_KEY)
# What each resolved channel is allowed to carry to the teacher on top of the
# required keys. The reference channel's privilege lives in the LATENT bundle
# and the caption channel's in the required hidden states themselves, so both
# carry nothing extra here; in particular neither may smuggle in an endpoint
# task, which would turn the teacher into a keyframe one under another name.
TEACHER_CHANNEL_KEYS: dict[str | None, tuple[str, ...]] = {
    "qwen": (H3_QWEN_CONTROL_VISUALS_KEY,),
    "keyframe": (H3_KEYFRAME_VISUALS_KEY, H3_CONDITIONING_TASK_KEY),
    "reference": (),
    "caption": (),
}

# The latent-side tensors that make up one Ref2VA reference bundle. They are
# matched by PREFIX because every one of them carries the reference sizing
# suffix (``mmh3_reference_kinds_se384``), which is part of the cache identity
# and not something this module is entitled to reconstruct. The sizing scalars
# that live beside them (short edge, size mode, temporal contract) are identity,
# not payload, and stay the student's: the two arms must agree on them, and a
# teacher cached with different sizing must fail the identity check rather than
# quietly override it.
REFERENCE_BUNDLE_KEYS = (
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
)

# How a teacher may be privileged. ``auto`` reads the caches and decides; the
# explicit values exist only to pin the channel when a corpus happens to carry
# more than one and the operator wants to measure a particular one.
TEACHER_PRIVILEGE_CHANNELS = ("auto", "qwen", "reference", "keyframe", "caption")

# The conditioning tasks that make a teacher an ENDPOINT teacher: it reads the
# clip's own first and/or last frames as conditioning, through the channel the
# checkpoint was distilled to use, while a ``t2va`` student reads none of them.
# This is the default teacher of kohya-ss/musubi-tuner PR #1047 -- there the
# student is text-only and the teacher is given ``first,last`` -- and it is the
# only channel here whose privilege reaches the model as clean latent rows
# rather than as pictures shown to the conditioner.
TEACHER_KEYFRAME_TASKS = ("i2va", "fl2va", "l2va")

# A window longer than this stops being a local correction: every sub-step past
# the first is taken under stop-grad from the student's own prediction, so the
# state drifts away from anything the teacher was measured on.
MAX_ROLLOUT_WINDOW = 3

# The supervision window may not walk the state into an exactly clean latent:
# sigma 0 leaves the flow field undefined for the next advance and gives the
# teacher a state no sampler ever evaluates.
SIGMA_FLOOR = 1e-3


def rollout_base_sigmas(stop_sigma: float, steps: int, window: int, *, floor: float = SIGMA_FLOOR) -> tuple[float, ...]:
    """Descending *unshifted* base sigmas for one rollout plus its window.

    ``steps`` uniform Euler steps carry the state from pure noise (base sigma 1,
    which is pure noise on every schedule because a shift maps 1 to 1) down to
    ``stop_sigma``. The supervision window then continues on the *same* step
    size, so a sub-step is exactly the step the sampler would have taken next --
    it is not a second, differently scaled schedule bolted onto the first.

    Returns ``steps + window`` values: index ``0`` is the noise state, index
    ``steps`` is the on-policy state the first supervision sub-step is taken at,
    and indices ``steps .. steps + window - 1`` are the supervised states. The
    advance from the last supervised state is never taken, so no extra entry is
    needed for it.
    """
    if steps < 1:
        raise ValueError("H3 rollout needs at least one Euler step")
    if not 1 <= window <= MAX_ROLLOUT_WINDOW:
        raise ValueError(f"H3 rollout window must lie in [1, {MAX_ROLLOUT_WINDOW}]")
    if not floor < stop_sigma < 1.0:
        raise ValueError(f"H3 rollout stop sigma must lie in ({floor}, 1), got {stop_sigma}")
    delta = (1.0 - stop_sigma) / steps
    sigmas = [1.0 - index * delta for index in range(steps)]
    # The stop is written from the drawn value rather than accumulated, so the
    # supervised state sits exactly at the sigma that was drawn.
    sigmas.append(stop_sigma)
    for index in range(1, window):
        # Clamped, not truncated: the window must not walk the state past sigma 0,
        # where the flow field is undefined for the next advance.
        #
        # KNOWN COST of clamping, reachable only above the default window of 1: when
        # the stride is larger than the stop sigma, several tail entries land on the
        # floor together. The advance between them spans a zero interval and leaves
        # the state untouched, so the same state is supervised more than once --
        # counted repeatedly in the loss and paid for with forwards that add no new
        # on-policy state. Truncating the tail there instead would fix that, but it
        # changes how many states the objective supervises, which is a change to the
        # objective and not to its bookkeeping; it wants its own measurement.
        sigmas.append(max(stop_sigma - index * delta, floor))
    return tuple(sigmas)


def euler_advance(state: torch.Tensor, velocity: torch.Tensor, sigma: float, next_sigma: float) -> torch.Tensor:
    """One Euler step of H3's flow, in that modality's own shifted sigma.

    With ``x_s = (1 - s) * x0 + s * noise`` and ``v = x0 - noise``, the state
    satisfies ``dx/ds = noise - x0 = -v``, so moving from ``s`` down to ``s'``
    adds ``(s - s') * v``. Taking ``s' = 0`` recovers the trainer's own
    ``x0 = x_t + (1 - t) * v`` with ``t = 1 - s``, which is the check that the
    two conventions are the same one.
    """
    if state.shape != velocity.shape:
        raise ValueError(f"H3 rollout state {tuple(state.shape)} and velocity {tuple(velocity.shape)} must match")
    if next_sigma > sigma:
        raise ValueError("H3 rollout Euler steps must descend in sigma")
    return state + (sigma - next_sigma) * velocity.to(dtype=state.dtype)


def is_reference_bundle_key(key: str) -> bool:
    """Whether a logical cache key is part of a Ref2VA reference bundle."""
    return any(key.startswith(prefix) for prefix in REFERENCE_BUNDLE_KEYS)


def _conditioning_task_id(batch: dict[str, Any]) -> int | None:
    """The task id a batch declares, through whichever varlen container holds it."""
    value = batch.get(H3_CONDITIONING_TASK_KEY)
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    return int(value)


def teacher_batch(
    batch: dict[str, Any],
    entries: dict[str, torch.Tensor],
    reference_entries: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Present the teacher's conditioning through the student's batch.

    The clone is shallow and replaces only the conditioning keys. Target latents,
    loss masks, geometry, the conditioning task and the packed layout stay the
    student's, so the two forwards differ in what the teacher was given and in
    nothing else -- which is exactly the difference the preflight measured.

    ``reference_entries``, when given, swaps the whole reference bundle for the
    teacher's. The bundle is replaced wholesale rather than merged: kinds, shapes
    and packed rows are cut apart against each other by the conditioning reader,
    so a bundle assembled from two caches would be internally inconsistent. The
    student's own bundle keys are dropped first, since the teacher's carry the
    reference count in their key-free contents and a stale student key would
    survive under a suffix the teacher does not use.

    Every replacement is wrapped as a one-element list, the varlen container the
    conditioning reader accepts for any key, so a teacher cache written with a
    different dtype suffix or varlen prefix than the student's still lands in the
    same slot.
    """
    clone = dict(batch)
    student_task = _conditioning_task_id(batch)
    for key, tensor in entries.items():
        clone[key] = [tensor]
    teacher_task = _conditioning_task_id(clone)
    # A student that declares no task at all is a stub batch, not a training one;
    # there is nothing to compare against and nothing to protect.
    if student_task is not None and teacher_task is not None and teacher_task != student_task:
        # An endpoint teacher differs from its student in this scalar alone, and
        # the difference is the privilege: the keyframe rows are already in the
        # shared batch, and the task is what decides whether a forward reads them
        # as conditioning. Everything else about the two presentations is equal by
        # construction, so nothing further has to be swapped to build it.
        expected = {H3_CONDITIONING_TASK_IDS[name] for name in TEACHER_KEYFRAME_TASKS}
        if teacher_task not in expected:
            task_by_id = {value: key for key, value in H3_CONDITIONING_TASK_IDS.items()}
            raise ValueError(
                "H3 rollout teacher declares conditioning task "
                f"{task_by_id.get(teacher_task, teacher_task)!r} where the student declares "
                f"{task_by_id.get(student_task, student_task)!r}; only an endpoint task "
                f"({', '.join(TEACHER_KEYFRAME_TASKS)}) may differ, since only those read the shared keyframe rows"
            )
    # A teacher without the control marker is a plain caption cache: under the
    # Qwen channel the whole premise of the objective would silently degrade into
    # self-distillation. Under the reference channel the marker is absent by
    # design, and dropping it is what keeps the presentations comparable.
    if H3_QWEN_CONTROL_VISUALS_KEY not in entries:
        clone.pop(H3_QWEN_CONTROL_VISUALS_KEY, None)
    if H3_KEYFRAME_VISUALS_KEY not in entries:
        clone.pop(H3_KEYFRAME_VISUALS_KEY, None)
    if reference_entries:
        for key in [key for key in clone if is_reference_bundle_key(key)]:
            del clone[key]
        for key, tensor in reference_entries.items():
            clone[key] = [tensor]
    return clone


class H3RolloutTeacherCache:
    """Item-keyed view of a second dataset's cached conditioning.

    The text side is always read. The latent side is opened only for the
    *reference bundle*: under the Qwen channel the teacher's latent cache is
    normally hard-linked from the student's -- control visuals never reach a VAE
    -- and nothing in it differs, while under the reference channel that cache is
    the entire privilege. The target latents of an active step always come from
    the ordinary batch, whichever channel is in use, because the rollout state
    the two arms are compared at must be one state.

    The tensors are read on demand behind a bounded cache, so a corpus of a few
    hundred clips costs a handful of megabytes rather than the whole cache
    resident.
    """

    def __init__(
        self,
        paths: dict[str, str],
        *,
        latent_paths: dict[str, str] | None = None,
        cache_size: int = 64,
    ) -> None:
        if not paths:
            raise ValueError("--h3_rollout_teacher_config resolved no cached text-encoder outputs")
        if cache_size < 1:
            raise ValueError(f"teacher cache size must be positive, got {cache_size}")
        self._paths = dict(paths)
        self._latent_paths = dict(latent_paths or {})
        self._cache_size = cache_size
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._reference_cache: dict[str, dict[str, torch.Tensor]] = {}
        # Memoised key resolution, and the sources already warned about, so a
        # teacher cached at one window does not print once per step.
        self._resolved: dict[str, str | None] = {}
        self._window_fallbacks: set[str] = set()
        self._path_index = windowed_key_index(self._paths)
        self._latent_index = windowed_key_index(self._latent_paths)
        # Resolved by ``validate_privilege``. Until then no step may run: a
        # teacher whose channel is unknown is a teacher nothing has checked.
        self._channel: str | None = None

    @classmethod
    def from_dataset_group(cls, dataset_group: Any, *, cache_size: int = 64) -> H3RolloutTeacherCache:
        """Map every item key of a prepared dataset group to its cache files."""
        return cls(
            item_key_text_caches(dataset_group),
            latent_paths=item_key_latent_caches(dataset_group),
            cache_size=cache_size,
        )

    @property
    def channel(self) -> str | None:
        """Which channel carries the privilege, once validation has decided."""
        return self._channel

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def item_keys(self) -> frozenset[str]:
        return frozenset(self._paths)

    def _resolve(self, item_key: str) -> str | None:
        """Which teacher key serves this student item, or ``None`` if none does.

        The exact window first, then the bare source key, then any other window
        of the same source. The last is a real substitution and says so once per
        source -- but a legitimate one: the teacher's privilege in variant B is
        its extra reference frames plus its text, and neither depends on how many
        frames of the clip the student's window covers. Refusing it would force
        the teacher corpus to mirror every entry of the student's ``target_frames``
        list for no gain in what the teacher actually presents.
        """
        if item_key in self._paths:
            return item_key
        if item_key in self._resolved:
            return self._resolved[item_key]
        chosen = resolve_windowed_key(item_key, self._paths, self._path_index)
        stem = item_key_stem(item_key)
        if chosen is not None and chosen != stem and stem not in self._window_fallbacks:
            self._window_fallbacks.add(stem)
            logger.warning(
                "H3 rollout teacher has no window %r; using %r for every window of that clip "
                "(the teacher's references and text do not depend on the window's length)",
                item_key,
                chosen,
            )
        self._resolved[item_key] = chosen
        return chosen

    def require(self, item_keys: Iterable[str]) -> None:
        """Fail loudly, before the transformer is loaded, on an unpaired corpus.

        Pairing is by rollout item key, which encodes the source clip, the crop
        position and the frame count -- so a teacher cache built at a different
        resolution does not merely mispair, it fails to pair at all, which is the
        outcome worth having. A teacher cached at a different ``target_frames``
        still pairs, per clip rather than per window; see :meth:`_resolve`.
        """
        missing = sorted(key for key in item_keys if self._resolve(key) is None)
        if missing:
            shown = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
            raise ValueError(
                "--h3_rollout_teacher_config must cache the SAME items as the training dataset; "
                f"{len(missing)} item(s) have no teacher entry: {shown}{more}. "
                "Rebuild the teacher cache from the same clips at the same resolution. "
                "(A teacher cached at other target_frames still pairs, per clip; these clips are absent entirely.)"
            )

    def validate_privilege(
        self,
        student_latent_paths: dict[str, str] | None = None,
        *,
        channel: str = "auto",
        student_text_paths: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """Check every teacher item is privileged somewhere; resolve the channel.

        Reads key names and reference counts. The one payload it does read is the
        Qwen control-visual COUNT, and only to reject a cache that declares the
        key with no visuals behind it -- see :func:`_qwen_payload_defect`. A
        teacher
        privileged in neither channel is the frozen base under the student's own
        conditioning, which turns the objective into self-distillation with no
        privileged signal at all -- a failure that would otherwise show up only
        as a training curve that went nowhere.

        Four channels count as privilege, and each is a difference between the
        two arms, checked against the student's own cache for the item whenever
        ``student_text_paths`` supplies one. ``qwen``: the teacher's text cache
        carries ``qwen_control_*`` visuals the student's does not present.
        ``reference``: the teacher's latent cache holds strictly MORE reference
        entries than the student's for the same item, which is the Ref2VA
        variant-B construction where extra target frames are declared as
        references. ``keyframe``: the teacher's text cache declares an endpoint
        conditioning task (``i2va``/``fl2va``/``l2va``) where the student's
        cache declares none, so the keyframe rows the shared latent cache always
        carries become conditioning for the teacher alone -- the endpoint teacher
        of PR #1047. A student cached with an endpoint task of its own reads the
        same rows, and the channel is then absent; without the student's text
        caches it cannot be assessed and is treated as absent too. ``caption``:
        the teacher's caption encodes to more tokens than the student's.

        Under ``auto`` the corpus decides. Keyframe wins because its privilege
        reaches the model as clean latent rows on the path the checkpoint was
        distilled to read, where the other two arrive as pictures shown to the
        conditioner or as references it may or may not attend to; between the
        remaining two, Qwen wins a tie because it is the channel the preflight
        measured.

        Returns the resolved channel and the number of paired items.
        """
        if channel not in TEACHER_PRIVILEGE_CHANNELS:
            raise ValueError(f"unknown H3 rollout teacher privilege channel {channel!r}")
        student_latent_paths = dict(student_latent_paths or {})
        student_text_paths = dict(student_text_paths or {})
        student_text_index = windowed_key_index(student_text_paths)
        student_index = windowed_key_index(student_latent_paths)
        endpoint_task_ids = {H3_CONDITIONING_TASK_IDS[name] for name in TEACHER_KEYFRAME_TASKS}
        without_qwen: list[str] = []
        degenerate_qwen: list[tuple[str, str]] = []
        without_reference: list[str] = []
        without_keyframe: list[str] = []
        without_caption: list[str] = []
        for item_key, path in sorted(self._paths.items()):
            with safe_open(path, framework="pt", device="cpu") as handle:
                logical = {logical_cache_key(key) for key in handle.keys()}
                task_key = next((key for key in handle.keys() if logical_cache_key(key) == H3_CONDITIONING_TASK_KEY), None)
                teacher_task = None if task_key is None else int(handle.get_tensor(task_key))
            missing = [key for key in TEACHER_REQUIRED_KEYS if key not in logical]
            if missing:
                raise ValueError(f"H3 teacher text cache {path} is missing {', '.join(missing)}")
            # Every channel is a DIFFERENCE between the two arms, so where the
            # student's text cache for the item is available each one is judged
            # against it: a teacher that merely HAS control visuals or an endpoint
            # task is not privileged when the student presents the same thing, and
            # that case -- both arms cached as fl2va, or both carrying the same
            # Qwen assets -- is self-distillation dressed up as a privileged
            # teacher. The trainer always supplies the student's caches. Without
            # them none of the three text-side channels can be assessed, and an
            # unassessable channel counts as absent: a missing baseline must never
            # make a teacher look privileged by default.
            student_text_key = resolve_windowed_key(item_key, student_text_paths, student_text_index)
            student_text_path = student_text_paths.get(student_text_key) if student_text_key else None
            student_logical, student_task = _student_presentation(student_text_path)
            if (
                H3_QWEN_CONTROL_VISUALS_KEY not in logical
                or student_logical is None
                or H3_QWEN_CONTROL_VISUALS_KEY in student_logical
            ):
                without_qwen.append(item_key)
            else:
                # Declared and present is not the same as informative. Read once,
                # here, where every teacher text cache is already open.
                defect = _qwen_payload_defect(path)
                if defect is not None:
                    degenerate_qwen.append((item_key, defect))
            if teacher_task not in endpoint_task_ids or student_logical is None or student_task in endpoint_task_ids:
                without_keyframe.append(item_key)
            teacher_references = reference_bundle_size(self._latent_paths.get(item_key))
            # The student is looked up through the same window resolution the
            # steps will use, so the count compared is the count that will be
            # substituted rather than an accidental zero from a missing window.
            student_key = resolve_windowed_key(item_key, student_latent_paths, student_index)
            student_references = reference_bundle_size(student_latent_paths.get(student_key) if student_key else None)
            if teacher_references <= student_references:
                without_reference.append(item_key)
            # Caption privilege: the teacher is shown a fuller description of the same
            # clip than the student is. Both sides stay text-only, so the gap between
            # them is something the student can actually close -- infer the detail a
            # longer caption would have supplied -- unlike Qwen visuals, which the
            # student has no access to at inference and cannot derive from its prompt.
            # Measured as the encoded length, because that is what the model sees; a
            # teacher whose caption encodes no longer than the student's carries nothing.
            #
            # Without the student's text caches there is nothing to be longer THAN, and
            # a zero baseline would call every non-empty teacher caption privileged --
            # silently, and ahead of the reference channel in the order below, so a
            # genuinely reference-privileged teacher would be reported as a caption one.
            # An absent baseline means the channel is unassessable, not satisfied.
            if not student_text_paths:
                without_caption.append(item_key)
            else:
                teacher_tokens = _caption_tokens(path)
                student_tokens = _caption_tokens(student_text_path) if student_text_path else 0
                if teacher_tokens <= student_tokens:
                    without_caption.append(item_key)
        if channel == "auto":
            if not without_keyframe:
                resolved = "keyframe"
            elif not without_qwen:
                resolved = "qwen"
            elif not without_caption:
                # Below the other two on purpose: caption privilege is the weakest
                # signal of the three and the easiest to create by accident, so it wins
                # only when nothing stronger is present.
                resolved = "caption"
            else:
                resolved = "reference"
        else:
            resolved = channel
        unprivileged = {
            "qwen": without_qwen,
            "reference": without_reference,
            "keyframe": without_keyframe,
            "caption": without_caption,
        }[resolved]
        if unprivileged:
            self._channel = None
            raise ValueError(
                self._privilege_error(
                    resolved, channel, unprivileged, without_qwen, without_reference, without_caption, without_keyframe
                )
            )
        if resolved == "qwen" and degenerate_qwen:
            self._channel = None
            shown = "; ".join(f"{key} {why}" for key, why in degenerate_qwen[:5])
            more = "" if len(degenerate_qwen) <= 5 else f" (and {len(degenerate_qwen) - 5} more)"
            raise ValueError(
                f"{len(degenerate_qwen)} of {len(self._paths)} teacher items declare qwen_control_* visuals that "
                f"carry no information: {shown}{more}. The channel resolves on key names, so such a corpus would "
                "otherwise train to completion with a teacher no better informed than the student. Re-cache the "
                "teacher arm with real, non-empty control visuals."
            )
        self._channel = resolved
        # Entries read before the channel was known carried every optional key.
        self._cache.clear()
        return resolved, len(self._paths)

    def _privilege_error(
        self,
        resolved: str,
        requested: str,
        unprivileged: Sequence[str],
        without_qwen: Sequence[str],
        without_reference: Sequence[str],
        without_caption: Sequence[str],
        without_keyframe: Sequence[str] = (),
    ) -> str:
        """Say which channel was looked at and what each one would have needed."""
        shown = ", ".join(unprivileged[:5])
        more = "" if len(unprivileged) <= 5 else f" (and {len(unprivileged) - 5} more)"
        fixes = {
            "qwen": (
                "Qwen teacher -- re-cache the teacher arm with real qwen_control_* assets in its TEXT cache"
            ),
            "reference": (
                "reference teacher -- run minimax_h3_prepare_rollout_teacher.py, then cache its extra target frames "
                "as references in a separate LATENT cache"
            ),
            "keyframe": (
                "endpoint teacher -- re-cache the teacher arm's TEXT outputs with --task fl2va while the student keeps "
                "its --task t2va cache; both arms share the latent cache, which already carries the keyframe rows"
            ),
        }
        if requested != "auto":
            others = "; or ".join(text for name, text in fixes.items() if name != resolved)
            return (
                f"--h3_rollout_teacher_privilege {requested} found {len(unprivileged)} unprivileged teacher item(s): "
                f"{shown}{more}. Either fix that channel, or switch channels: {others}."
            )
        # Under auto every channel was measured, so every count is known and
        # naming only the losing one would hide most of the diagnosis.
        return (
            f"{len(unprivileged)} teacher item(s) carry no privilege over the student: {shown}{more}. "
            f"{len(without_caption)} item(s) carry no longer caption than the student's, "
            f"{len(without_qwen)} item(s) carry no Qwen control visuals, {len(without_reference)} item(s) carry no "
            f"extra reference latents and {len(without_keyframe)} item(s) declare no endpoint conditioning task, so the "
            "rollout would be self-distillation with no privileged signal at all. "
            "Choose a channel: " + "; or ".join(fixes.values()) + "."
        )

    def entries(self, item_key: str) -> dict[str, torch.Tensor]:
        """The text-side tensors of one teacher item, keyed logically."""
        resolved_key = self._resolve(item_key)
        path = self._paths.get(resolved_key) if resolved_key else None
        if path is None:
            raise KeyError(
                f"H3 rollout teacher has no entry for item {item_key!r}; "
                "the teacher config must cache the same items as the training dataset"
            )
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        entries: dict[str, torch.Tensor] = {}
        # Only the resolved channel's presentation travels to the teacher. A cache
        # that carries several (Qwen assets AND an endpoint task, say) would
        # otherwise hand the teacher every one of them whatever channel was chosen,
        # and an ablation labelled "caption" would train against a Qwen-informed
        # teacher. Before validation resolves a channel every optional key is
        # carried, so a teacher read without validation is the old permissive one.
        allowed = set(TEACHER_REQUIRED_KEYS) | set(TEACHER_CHANNEL_KEYS.get(self._channel, TEACHER_OPTIONAL_KEYS))
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                logical = logical_cache_key(key)
                if logical in allowed:
                    if logical in entries:
                        raise ValueError(f"H3 teacher text cache {path} carries two {logical} tensors")
                    entries[logical] = handle.get_tensor(key)
        missing = [key for key in TEACHER_REQUIRED_KEYS if key not in entries]
        if missing:
            raise ValueError(f"H3 teacher text cache {path} is missing {', '.join(missing)}")
        if len(self._cache) >= self._cache_size:
            # Arbitrary eviction: items arrive in the dataset's shuffled order, so
            # no recency order carries information worth the bookkeeping.
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = entries
        return entries

    def reference_entries(self, item_key: str) -> dict[str, torch.Tensor]:
        """The teacher's Ref2VA reference bundle, or empty under the Qwen channel.

        Empty is a legal answer, not a failure: on the Qwen channel the teacher's
        references are the student's references, and substituting them would be a
        no-op that only risks disagreeing about the sizing suffix. The bundle is
        returned with its suffixed keys intact, exactly as the conditioning
        reader looks them up.
        """
        if self._channel != "reference":
            return {}
        # Resolved against the latent map itself rather than through the text
        # side: the two maps are built from one dataset group and agree, and the
        # warning about a substituted window has already been printed there.
        latent_key = resolve_windowed_key(item_key, self._latent_paths, self._latent_index)
        path = self._latent_paths.get(latent_key) if latent_key else None
        if path is None:
            raise KeyError(
                f"H3 rollout teacher has no latent cache for item {item_key!r}; "
                "the reference channel needs the teacher dataset's own latents"
            )
        cached = self._reference_cache.get(path)
        if cached is not None:
            return cached
        entries: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                logical = logical_cache_key(key)
                if not is_reference_bundle_key(logical):
                    continue
                if logical in entries:
                    raise ValueError(f"H3 teacher latent cache {path} carries two {logical} tensors")
                entries[logical] = handle.get_tensor(key)
        if not entries:
            raise ValueError(
                f"H3 teacher latent cache {path} carries no reference bundle, but the rollout teacher was "
                "validated on the reference channel; the teacher cache was rebuilt underneath the run"
            )
        if len(self._reference_cache) >= self._cache_size:
            self._reference_cache.pop(next(iter(self._reference_cache)))
        self._reference_cache[path] = entries
        return entries


def reference_bundle_size(path: str | None) -> int:
    """How many references a latent cache holds, read from its kinds vector.

    The kinds vector has one entry per reference and is what the conditioning
    reader cuts the packed rows apart by, so its length is the reference count in
    the only sense the model cares about. A missing path or a cache without a
    bundle counts as zero rather than raising: a T2VA student legitimately has
    none, and "the teacher has more" is the question being asked.
    """
    if path is None:
        return 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            logical = logical_cache_key(key)
            if logical.startswith(H3_REFERENCE_KINDS_KEY):
                return int(handle.get_tensor(key).numel())
    return 0


def rollout_item_key(item: Any) -> str:
    """The key one dataset item is paired by: its source key plus its window.

    ``item_key`` alone names the SOURCE, not the item: a ``target_frames`` list
    cuts several windows out of one clip and every window keeps the clip's key,
    so keying by it collapses them onto each other. The windowed key is the stem
    of the item's own cache files, which is the finest identity the caches have.
    """
    return getattr(item, "windowed_item_key", None) or item.item_key


def item_key_stem(item_key: str) -> str:
    """A rollout item key with its window token removed, i.e. the source key."""
    return _WINDOW_SUFFIX.sub("", item_key)


def windowed_key_index(available: Iterable[str]) -> dict[str, str]:
    """``source key -> the first of its windows``, for repeated resolution.

    Built once by callers that resolve many keys against one map; without it each
    miss would rescan the whole map.
    """
    index: dict[str, str] = {}
    for key in sorted(available):
        index.setdefault(item_key_stem(key), key)
    return index


def resolve_windowed_key(item_key: str, available: Mapping[str, Any], index: Mapping[str, str] | None = None) -> str | None:
    """Which key of ``available`` serves ``item_key``, exact window or not.

    Three answers, in order. The same window, which is what a teacher cached from
    the same ``target_frames`` list gives. The bare source key, which is what a
    map built before windows were distinguished at all gives -- accepted so that
    single-window teacher corpora cached by older code keep pairing. Otherwise
    another window of the same source, which callers should warn about.

    Returns ``None`` when the source is absent entirely, which is the failure
    ``require`` reports.
    """
    if item_key in available:
        return item_key
    stem = item_key_stem(item_key)
    if stem in available:
        return stem
    if index is None:
        index = windowed_key_index(available)
    return index.get(stem)


def _item_key_caches(dataset_group: Any, attribute: str, label: str) -> dict[str, str]:
    """Collect ``rollout item key -> cache path`` from a prepared group's buckets.

    Read from the batch managers' buckets rather than reconstructed from a
    directory glob: the bucket contents are what ``__getitem__`` will actually
    load, so the mapping is exact. ``num_repeats`` puts the same item in a bucket
    several times, which is why duplicates are collapsed rather than rejected.
    """
    paths: dict[str, str] = {}
    for dataset in getattr(dataset_group, "datasets", []):
        manager = getattr(dataset, "batch_manager", None)
        if manager is None:
            continue
        for bucket in manager.buckets.values():
            for item in bucket:
                path = getattr(item, attribute, None)
                if path is None:
                    continue
                resolved = os.path.abspath(path)
                key = rollout_item_key(item)
                previous = paths.get(key)
                if previous is not None and previous != resolved:
                    raise ValueError(f"H3 item {key!r} resolves to two {label} caches: {previous} and {resolved}")
                paths[key] = resolved
    return paths


def _student_presentation(path: str | None) -> tuple[set[str] | None, int | None]:
    """The logical keys a student text cache carries, and its conditioning task.

    ``(None, None)`` when there is no student cache to read, which every caller
    treats as "cannot be assessed" rather than as "the student presents nothing":
    a missing baseline must never make the teacher look privileged by default.
    A cache without a task tensor declares the text-only ``t2va`` presentation,
    returned as ``None`` so it compares unequal to every endpoint task id.
    """
    if not path:
        return None, None
    with safe_open(path, framework="pt", device="cpu") as handle:
        logical = {logical_cache_key(key) for key in handle.keys()}
        task_key = next((key for key in handle.keys() if logical_cache_key(key) == H3_CONDITIONING_TASK_KEY), None)
        task = None if task_key is None else int(handle.get_tensor(task_key))
    return logical, task


def _qwen_payload_defect(path: str) -> str | None:
    """Why this item's control-visual marker declares nothing, or ``None``.

    The marker under ``H3_QWEN_CONTROL_VISUALS_KEY`` is not the pictures: the
    conditioner already folded those into the cached hidden states, and what the
    cache keeps is the COUNT of visuals it was shown, as one integer scalar
    (``conditioning.py`` writes ``torch.tensor(len(qwen_controls))``). So the
    only payload question a validator can answer is whether that count is
    positive. A cache that declares the key with an empty tensor or a count of
    zero passes every name-based check, resolves to the Qwen channel, and trains
    to completion against a teacher that saw nothing the student did not; the
    training curve is the only symptom, and it reads as a weak method.
    """
    with safe_open(path, framework="pt", device="cpu") as handle:
        key = next((name for name in handle.keys() if logical_cache_key(name) == H3_QWEN_CONTROL_VISUALS_KEY), None)
        if key is None:
            return None
        count = handle.get_tensor(key)
    if count.numel() == 0:
        return "declares control visuals with an empty count tensor"
    if int(count.reshape(-1)[0]) <= 0:
        return "declares zero control visuals"
    return None


def _caption_tokens(path: str | None) -> int:
    """Encoded caption length in a text cache, or 0 when there is nothing to read.

    Read from the tensor shape rather than the caption string, because the length that
    matters is the one the model is presented with after encoding and truncation, not
    the one in the dataset file.
    """
    if not path:
        return 0
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            key = next((name for name in handle.keys() if logical_cache_key(name) == H3_TEXT_HIDDEN_KEY), None)
            if key is None:
                return 0
            shape = handle.get_slice(key).get_shape()
    except (OSError, ValueError):
        return 0
    # (..., tokens, width): the token axis is the second from the end.
    return int(shape[-2]) if len(shape) >= 2 else 0


def item_key_text_caches(dataset_group: Any) -> dict[str, str]:
    """``item_key -> text-encoder cache path`` for a prepared dataset group."""
    return _item_key_caches(dataset_group, "text_encoder_output_cache_path", "text")


def item_key_latent_caches(dataset_group: Any) -> dict[str, str]:
    """``item_key -> latent cache path`` for a prepared dataset group.

    The reference bundle lives here, which is what makes this the second half of
    the pairing under variant B: the same item key that names a text cache names
    the latents whose reference count decides whether the teacher is privileged.
    """
    return _item_key_caches(dataset_group, "latent_cache_path", "latent")


def enable_item_keys(dataset_group: Any, key: str = H3_ROLLOUT_ITEM_KEYS_BATCH_KEY) -> None:
    """Ask every batch manager of a group to name the items it emits.

    The batch dict is otherwise identity-free, and the rollout has to look up a
    teacher entry for the item it is training on. Passing the key name in rather
    than importing a constant into the dataset layer keeps ``bucket.py`` free of
    any H3 dependency.
    """
    for dataset in getattr(dataset_group, "datasets", []):
        manager = getattr(dataset, "batch_manager", None)
        if manager is not None:
            manager.h3_item_keys_batch_key = key


def batch_item_key(batch: dict[str, Any]) -> str:
    """The single item key behind a batch-size-1 H3 step."""
    keys: Sequence[str] | None = batch.get(H3_ROLLOUT_ITEM_KEYS_BATCH_KEY)
    if keys is None:
        raise KeyError(
            f"--h3_rollout_supervision needs the dataset to name its items; the batch carries no {H3_ROLLOUT_ITEM_KEYS_BATCH_KEY}"
        )
    if isinstance(keys, str):
        return keys
    if len(keys) != 1:
        raise ValueError("H3 rollout supervision runs one item per step")
    return str(keys[0])
