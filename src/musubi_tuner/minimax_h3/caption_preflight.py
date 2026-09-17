"""Caption-only H3 validation, performed before loading a text encoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptionIssue:
    dataset_index: int
    item_key: str
    caption_path: str
    kind: str
    detail: str


@dataclass
class CaptionPreflightReport:
    checked: int = 0
    issues: list[CaptionIssue] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"H3 caption preflight: {len(self.issues)} problem(s) in {self.checked} item(s)."]
        for issue in self.issues:
            lines.append(
                f"  dataset {issue.dataset_index + 1}: {issue.item_key} -> {issue.caption_path}: {issue.kind} ({issue.detail})"
            )
        lines.append("Add or fix these captions before caching H3 conditioning.")
        return "\n".join(lines)


def _caption_location(source: Any, key: str, index: int) -> str:
    for name in ("image_jsonl_file", "video_jsonl_file", "audio_jsonl_file"):
        if getattr(source, name, None):
            return f"{getattr(source, name)} (record {index + 1}, caption)"
    extension = getattr(source, "caption_extension", None)
    if extension:
        return str(getattr(source, "caption_paths", {}).get(key, str(Path(key).with_suffix(extension))))
    return f"{key} (caption)"


def _hidden_images(source: Any) -> list[str]:
    if not getattr(source, "image_directory", None) or not getattr(source, "caption_extension", None):
        return []
    from musubi_tuner.dataset.media_utils import glob_images

    existing = set(source.image_paths)
    companions = {path for paths in getattr(source, "target_paths", {}).values() for path in paths}
    return [path for path in glob_images(source.image_directory) if path not in existing and path not in companions]


def scan_caption_preflight(dataset_group: Any) -> CaptionPreflightReport:
    """Read every caption without opening media, preserving datasource iteration state."""
    report = CaptionPreflightReport()
    for dataset_index, dataset in enumerate(dataset_group.datasets):
        source = getattr(dataset, "datasource", None)
        hidden = _hidden_images(source) if source is not None else []
        if source is None and not hasattr(dataset, "records"):
            raise ValueError(f"H3 caption preflight does not support dataset {dataset_index + 1}: no caption datasource")
        count = len(source) if source is not None else len(dataset.records)
        for index in range(count):
            key = str(source.get_item_key(index)) if source is not None else str(dataset.records[index][0])
            if source is not None and hasattr(source, "should_include_item") and not source.should_include_item(key):
                continue
            report.checked += 1
            location = _caption_location(source if source is not None else dataset, key, index)
            try:
                _, caption = source.get_caption(index) if source is not None else dataset.records[index]
                if caption is None:
                    kind, detail = "missing", "caption is null"
                elif not isinstance(caption, str):
                    kind, detail = "non-string", f"expected string, got {type(caption).__name__}"
                elif not caption.strip():
                    kind, detail = "empty", "caption contains no non-whitespace text"
                else:
                    continue
            except (FileNotFoundError, KeyError) as error:
                kind, detail = "missing", str(error)
            except (OSError, UnicodeError) as error:
                kind, detail = "unreadable", f"{type(error).__name__}: {error}"
            report.issues.append(CaptionIssue(dataset_index, key, location, kind, detail))
        for key in hidden:
            report.checked += 1
            report.issues.append(
                CaptionIssue(
                    dataset_index,
                    key,
                    _caption_location(source, key, -1),
                    "missing",
                    "media was excluded because its caption sidecar is missing",
                )
            )
    return report


def require_caption_preflight(dataset_group: Any, report: CaptionPreflightReport | None = None) -> CaptionPreflightReport:
    """Reject invalid captions without changing datasource records or fetchers."""
    report = report or scan_caption_preflight(dataset_group)
    if report.issues:
        raise ValueError(report.summary())
    return report
