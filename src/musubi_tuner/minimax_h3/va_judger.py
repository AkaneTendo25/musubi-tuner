"""VA-Judger pairwise reward client and NFT objective for H3.

Protocol and objective provenance: ShareLab-SII/VA-Judger commit
``abe2ea0a63a55c2ddb2f0960d78a74a1a707c829``.  The released repository does
not contain a license file, so this module is a clean, small interoperability
implementation rather than a copied source file.

H3 predicts a clean sample with ``x0 = xt + sigma * v``.  Callers pass that
reconstructed clean prediction as ``pred_x0`` so the mixture operates in the
same target space as the released implementation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch


SCORE_DIMENSIONS = ("A", "B", "C", "D", "E")
_UPSTREAM_SHA = "abe2ea0a63a55c2ddb2f0960d78a74a1a707c829"
_Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


class VAJudgerError(RuntimeError):
    """A reward request failed or returned an unusable score."""


def _http_post_json(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - endpoint is supplied by the user
            status = response.status
            body = response.read()
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise VAJudgerError(f"VA-Judger returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise VAJudgerError(f"VA-Judger request failed: {exc}") from exc
    if status != 200:
        raise VAJudgerError(f"VA-Judger returned HTTP {status}")
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VAJudgerError("VA-Judger response is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise VAJudgerError("VA-Judger response must be a JSON object")
    return decoded


class VAJudgerClient:
    """Client for the released Qwen3-Omni pair reward server.

    ``score_group`` evaluates every unordered pair in both orientations, then
    maps scores back to candidate identity and averages each candidate's raw
    1--10 score per A--E dimension over its ``2 * (K - 1)`` appearances.  This
    cancels systematic video-1/video-2 position preference.  The server and
    training process therefore need shared access to the video paths.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 900.0,
        pair_batch_size: int = 8,
        transport: _Transport | None = None,
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if endpoint.endswith("/predict"):
            endpoint = endpoint[: -len("/predict")]
        if not endpoint:
            raise ValueError("VA-Judger endpoint must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("VA-Judger timeout must be positive and finite")
        if pair_batch_size <= 0:
            raise ValueError("VA-Judger pair_batch_size must be positive")
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.pair_batch_size = int(pair_batch_size)
        self._transport = transport or _http_post_json

    def score_group(self, prompt: str, video_paths: Sequence[str | Path]) -> torch.Tensor:
        """Return raw dimension scores with shape ``[K, 5]`` in A--E order."""

        prompt = str(prompt).strip()
        paths = [str(path) for path in video_paths]
        if not prompt:
            raise ValueError("VA-Judger prompt must not be empty")
        if len(paths) < 2:
            raise ValueError("VA-Judger group scoring requires at least two videos")
        if any(not path.strip() for path in paths):
            raise ValueError("VA-Judger video paths must not be empty")

        pairs: list[dict[str, Any]] = []
        indices: list[tuple[int, int]] = []
        pair_count = len(paths) * (len(paths) - 1)
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                for first, second in ((i, j), (j, i)):
                    pair_index = len(indices)
                    pairs.append(
                        {
                            "id": f"group0000_pair{pair_index:04d}_{first}_{second}",
                            "prompt": prompt,
                            "video_1": paths[first],
                            "video_2": paths[second],
                            "group_index": 0,
                            "group_size": len(paths),
                            "group_pair_count": pair_count,
                            "pair_index_in_group": pair_index,
                            "sample_index_1": first,
                            "sample_index_2": second,
                        }
                    )
                    indices.append((first, second))

        totals = torch.zeros((len(paths), len(SCORE_DIMENSIONS)), dtype=torch.float64)
        counts = torch.zeros(len(paths), dtype=torch.int64)
        for start in range(0, len(pairs), self.pair_batch_size):
            batch = pairs[start : start + self.pair_batch_size]
            response = self._transport(f"{self.endpoint}/predict", {"pairs": batch}, self.timeout)
            results = _parse_response(response, batch)
            for (i, j), result in zip(indices[start : start + len(batch)], results, strict=True):
                first, second = _parse_dimension_scores(result)
                totals[i] += first
                totals[j] += second
                counts[i] += 1
                counts[j] += 1

        expected = 2 * (len(paths) - 1)
        if not bool(torch.all(counts == expected)):
            raise VAJudgerError(f"incomplete all-pair aggregation: counts={counts.tolist()}, expected={expected}")
        return (totals / counts[:, None]).to(torch.float32)


def _parse_response(response: Mapping[str, Any], expected_pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if response.get("status") != "success":
        raise VAJudgerError(f"VA-Judger did not report success: {response.get('status')!r}")
    results = response.get("results")
    expected = len(expected_pairs)
    if not isinstance(results, list) or len(results) != expected:
        actual = len(results) if isinstance(results, list) else type(results).__name__
        raise VAJudgerError(f"VA-Judger returned {actual} results for {expected} pairs")
    if not all(isinstance(result, Mapping) for result in results):
        raise VAJudgerError("VA-Judger pair results must be JSON objects")
    for result, pair in zip(results, expected_pairs, strict=True):
        if result.get("id") != pair["id"]:
            raise VAJudgerError(f"VA-Judger result order/identity mismatch: expected {pair['id']!r}, got {result.get('id')!r}")
    return results


def _parse_dimension_scores(result: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = result.get("dimension_scores")
    if not isinstance(dimensions, Mapping):
        raise VAJudgerError(
            "VA-Judger result is missing dimension_scores; launch the official server with --prompt_mode dimension_scores"
        )
    first: list[float] = []
    second: list[float] = []
    for dimension in SCORE_DIMENSIONS:
        values = dimensions.get(dimension)
        if not isinstance(values, Mapping) or set(values) != {"1", "2"}:
            raise VAJudgerError(f"VA-Judger dimension {dimension} must contain exactly scores 1 and 2")
        for side, destination in (("1", first), ("2", second)):
            value = values[side]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise VAJudgerError(f"VA-Judger dimension {dimension} score {side} is not numeric")
            score = float(value)
            if not math.isfinite(score) or not 1.0 <= score <= 10.0:
                raise VAJudgerError(f"VA-Judger dimension {dimension} score {side} is outside [1, 10]")
            destination.append(score)
    if set(dimensions) != set(SCORE_DIMENSIONS):
        raise VAJudgerError("VA-Judger dimension_scores must contain exactly A, B, C, D, and E")
    return torch.tensor(first, dtype=torch.float64), torch.tensor(second, dtype=torch.float64)


def dimension_normalized_advantages(scores: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Standardize each reward dimension across a group, then average dimensions.

    This is the shared-scalar GDPO construction from the released trainer: a
    high-variance score dimension cannot dominate merely because of its scale.
    H3 applies the resulting scalar to both audio and video objectives.
    """

    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 1:
        raise ValueError("scores must have shape [group_size >= 2, dimensions >= 1]")
    if not scores.is_floating_point():
        scores = scores.float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    normalized = (scores - scores.mean(dim=0, keepdim=True)) / (scores.std(dim=0, correction=0, keepdim=True) + eps)
    return normalized.mean(dim=1)


def reward_components(scores: torch.Tensor) -> torch.Tensor:
    """Map raw A--E scores to upstream overall, audio, and video components.

    Output order is ``overall, audio, video``.  Normalize these three columns
    with :func:`dimension_normalized_advantages`; normalizing all five raw
    dimensions would define a different objective from released VA-Judger.
    """

    if scores.ndim != 2 or scores.shape[1] != len(SCORE_DIMENSIONS):
        raise ValueError("raw VA-Judger scores must have shape [group_size, 5]")
    if not scores.is_floating_point():
        scores = scores.float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("raw VA-Judger scores must be finite")
    overall = (scores[:, 0] + scores[:, 1] + scores[:, 4]) / 30.0
    audio = scores[:, 2] / 10.0
    video = scores[:, 3] / 10.0
    return torch.stack((overall, audio, video), dim=1)


def nft_mixture_loss(
    pred_x0: torch.Tensor,
    old_x0: torch.Tensor,
    reference_x0: torch.Tensor,
    target_x0: torch.Tensor,
    advantage: torch.Tensor | float,
    *,
    beta_mix: float = 1.0,
    kl_beta: float = 1e-4,
    advantage_clip: float = 5.0,
    element_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the released NFT positive/negative mixture plus reference MSE.

    Defaults reproduce ``videorl/scripts/train.py`` at the pinned upstream SHA.
    The old policy, reference, clean target, advantage, and mask are treated as
    fixed targets.  The L1 normalizers remain differentiable with respect to
    ``pred_x0``, exactly as in the source objective.
    """

    if pred_x0.shape != old_x0.shape or pred_x0.shape != reference_x0.shape or pred_x0.shape != target_x0.shape:
        raise ValueError("pred_x0, old_x0, reference_x0, and target_x0 must have identical shapes")
    if pred_x0.ndim < 1 or pred_x0.shape[0] < 1:
        raise ValueError("x0 tensors must include a nonempty batch dimension")
    for name, value, strictly_positive in (
        ("beta_mix", beta_mix, True),
        ("kl_beta", kl_beta, False),
        ("advantage_clip", advantage_clip, True),
    ):
        if not math.isfinite(value) or (value <= 0 if strictly_positive else value < 0):
            relation = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{name} must be {relation} and finite")

    old = old_x0.detach().to(device=pred_x0.device, dtype=pred_x0.dtype)
    reference = reference_x0.detach().to(device=pred_x0.device, dtype=pred_x0.dtype)
    target = target_x0.detach().to(device=pred_x0.device, dtype=pred_x0.dtype)
    adv = torch.as_tensor(advantage, device=pred_x0.device, dtype=pred_x0.dtype).detach()
    if adv.numel() == 1:
        adv = adv.expand(pred_x0.shape[0])
    if adv.shape != (pred_x0.shape[0],):
        raise ValueError(f"advantage must be scalar or shape [{pred_x0.shape[0]}]")

    positive = beta_mix * pred_x0 + (1.0 - beta_mix) * old
    negative = (1.0 + beta_mix) * old - beta_mix * pred_x0
    mask = None
    if element_mask is not None:
        mask = element_mask.detach().to(device=pred_x0.device, dtype=pred_x0.dtype)
        try:
            mask = torch.broadcast_to(mask, pred_x0.shape)
        except RuntimeError as exc:
            raise ValueError("element_mask is not broadcastable to x0") from exc
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
            raise ValueError("element_mask must be finite and nonnegative")

    positive_error = positive - target
    negative_error = negative - target
    norm_mask = mask.double() if mask is not None else None
    wp = _per_sample_mean(positive_error.double().abs(), norm_mask).clamp_min(1e-5)
    wn = _per_sample_mean(negative_error.double().abs(), norm_mask).clamp_min(1e-5)
    positive_term = _per_sample_mean(positive_error.square(), mask) / wp
    negative_term = _per_sample_mean(negative_error.square(), mask) / wn
    reward_mix = ((adv.clamp(-advantage_clip, advantage_clip) / advantage_clip) / 2.0 + 0.5).clamp(0.0, 1.0)
    policy = advantage_clip / beta_mix * (reward_mix * positive_term + (1.0 - reward_mix) * negative_term)
    reference_mse = _per_sample_mean((pred_x0 - reference).square(), mask)
    return (policy + kl_beta * reference_mse).mean()


def _per_sample_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    flat = values.reshape(values.shape[0], -1)
    if mask is None:
        return flat.mean(dim=1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    denominator = flat_mask.sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("element_mask must select at least one element per sample")
    return (flat * flat_mask).sum(dim=1) / denominator


__all__ = [
    "SCORE_DIMENSIONS",
    "VAJudgerClient",
    "VAJudgerError",
    "dimension_normalized_advantages",
    "nft_mixture_loss",
    "reward_components",
]
