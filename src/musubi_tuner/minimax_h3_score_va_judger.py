from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from musubi_tuner.minimax_h3.va_judger import SCORE_DIMENSIONS, VAJudgerClient, VAJudgerError

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DIM = re.compile(r"^\s*\(\s*([A-E])\s*\)", re.I | re.M)
_VIDEO = re.compile(r"video\s*([12])\s*(?:[:(]\s*|(?=\d+(?:\.\d+)?\s*/\s*10))", re.I)
_LABELS = {
    "A": "prompt adherence",
    "B": "audio-video alignment",
    "C": "audio quality",
    "D": "video quality",
    "E": "completeness and coherence",
}

# Behavior-level interoperability reference: ShareLab-SII/VA-Judger at
# abe2ea0a63a55c2ddb2f0960d78a74a1a707c829. This rubric is independently authored.
SCORING_PROMPT_VERSION = "h3-va-judger-ae-v1"
SCORING_IMPLEMENTATION_VERSION = "native-av-uniform-kv-bidir-v4"
SCORING_PROMPT = """Judge two generated audio-video clips against the supplied caption. Score both clips from 1 to 10 on every dimension:
(A) Prompt adherence, including requested characters, content, actions, and speech.
(B) Audio-video alignment, including lip synchronization and whether sounds fit the visible scene.
(C) Audio quality, including clarity, intelligibility, and freedom from distortion or noise.
(D) Video quality, including visual clarity, temporal stability, and freedom from artifacts.
(E) Completeness and coherence of the clip as a whole.

Inside <think>...</think>, give a concise rationale for each score and use exactly one line per dimension in this form:
(A) video 1: N/10 - rationale; video 2: N/10 - rationale
Continue identically for (B), (C), (D), and (E), then state the summed total for each video.

Finish with exactly one of these lines and no text after it:
<answer>video 1 is better</answer>
<answer>video 2 is better</answer>"""
SCORING_PROMPT_SHA256 = hashlib.sha256(SCORING_PROMPT.encode("utf-8")).hexdigest()


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score H3 rollouts with native Transformers VA-Judger")
    p.add_argument("--round_manifest", type=Path, required=True)
    p.add_argument("--reward_model", type=Path, required=True)
    p.add_argument("--pair_batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=12)
    p.add_argument("--video_max_pixels", type=int, default=602112)
    p.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be int")
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid round manifest: {exc}") from exc
    fields = {
        "version",
        "round_id",
        "start_step",
        "prompt_cursor",
        "group_count",
        "behavior_sha256",
        "semantic_fingerprint",
        "resume_path",
        "resume_sha256",
        "groups",
    }
    if not isinstance(data, dict) or set(data) != fields or data["version"] != 1:
        raise ValueError("invalid round manifest version or fields")
    for key in ("round_id", "behavior_sha256", "semantic_fingerprint"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"round manifest {key} must be nonempty")
    groups = data["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("round manifest groups must be nonempty")
    for key in ("start_step", "prompt_cursor", "group_count"):
        if _integer(data[key], key) < 0:
            raise ValueError(f"round manifest {key} must be nonnegative")
    if data["group_count"] != len(groups):
        raise ValueError("round manifest group_count is inconsistent")
    if not isinstance(data["resume_path"], str) or not Path(data["resume_path"]).is_absolute():
        raise ValueError("round manifest resume_path must be absolute")
    if not isinstance(data["resume_sha256"], str) or not _SHA.fullmatch(data["resume_sha256"]):
        raise ValueError("round manifest resume_sha256 is invalid")
    seen = set()
    for number, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != {"group_id", "prompt", "candidates"}:
            raise ValueError(f"group {number} has unexpected fields")
        gid = _integer(group["group_id"], f"group {number} group_id")
        if gid in seen:
            raise ValueError(f"duplicate group_id {gid}")
        seen.add(gid)
        if not isinstance(group["prompt"], str) or not group["prompt"].strip():
            raise ValueError(f"group {gid} prompt must be nonempty")
        if not isinstance(group["candidates"], list) or len(group["candidates"]) < 2:
            raise ValueError(f"group {gid} must have at least two candidates")
        for index, candidate in enumerate(group["candidates"]):
            label = f"group {gid} candidate {index}"
            required = {"path", "latent_path", "seed", "media_sha256", "latent_sha256"}
            if not isinstance(candidate, dict) or set(candidate) != required:
                raise ValueError(f"{label} has unexpected fields")
            _integer(candidate["seed"], f"{label} seed")
            for key in ("path", "latent_path"):
                if not isinstance(candidate[key], str) or not Path(candidate[key]).is_absolute():
                    raise ValueError(f"{label} {key} must be absolute")
            for key in ("media_sha256", "latent_sha256"):
                if not isinstance(candidate[key], str) or not _SHA.fullmatch(candidate[key]):
                    raise ValueError(f"{label} {key} is invalid")
    return data, raw


def _verify(manifest: Mapping[str, Any], latents: bool) -> None:
    if latents:
        resume = Path(manifest["resume_path"])
        if not resume.is_file() or _sha256(resume) != manifest["resume_sha256"]:
            raise ValueError(f"artifact hash mismatch: {resume}")
    for group in manifest["groups"]:
        for candidate in group["candidates"]:
            checks = [("path", "media_sha256")]
            if latents:
                checks.append(("latent_path", "latent_sha256"))
            for path_key, hash_key in checks:
                path = Path(candidate[path_key])
                if not path.is_file():
                    raise FileNotFoundError(f"artifact not found: {path}")
                if _sha256(path) != candidate[hash_key]:
                    raise ValueError(f"artifact hash mismatch: {path}")


def _parse_scores(text: str) -> dict[str, dict[str, float]]:
    matches = list(_DIM.finditer(text))
    result: dict[str, dict[str, float]] = {}
    for index, match in enumerate(matches):
        dimension = match.group(1).upper()
        if dimension in result:
            raise VAJudgerError(f"native output repeats dimension {dimension}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end() : end]
        segment = re.split(r"\n\s*(?:(?:Summed\s+)?Totals?\b|</think>|<answer>)", segment, maxsplit=1, flags=re.I)[0]
        markers = list(_VIDEO.finditer(segment))
        if len(markers) == 2 and {m.group(1) for m in markers} == {"1", "2"}:
            slots = [
                (m.group(1), segment[m.end() : markers[i + 1].start() if i + 1 < len(markers) else len(segment)])
                for i, m in enumerate(markers)
            ]
        elif len(markers) == 1 and markers[0].group(1) == "2":
            # Some native answers omit only the first video label. Its score must
            # start the dimension; the second must still identify video 2.
            slots = [("1", segment[: markers[0].start()]), ("2", segment[markers[0].end() :])]
        else:
            raise VAJudgerError(f"native output has ambiguous {dimension} video labels")
        values = {}
        pattern = rf"^\s*(?:{re.escape(_LABELS[dimension])}\s*:\s*)?(\d+(?:\.\d+)?)\s*/\s*10\b"
        for video, slot in slots:
            found = re.search(pattern, slot, re.I)
            if found is None:
                raise VAJudgerError(f"native output is missing {dimension} video {video} score")
            value = float(found.group(1))
            if not math.isfinite(value) or not 1 <= value <= 10:
                raise VAJudgerError(f"native score {dimension}/{video} is outside [1, 10]")
            values[video] = value
        result[dimension] = values
    if set(result) != set(SCORE_DIMENSIONS):
        raise VAJudgerError("native output must contain exactly dimensions A, B, C, D, and E")
    return result


def _load_video_audio(path: str | Path, sampling_rate: int):
    import av
    import numpy as np

    if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise ValueError("audio sampling_rate must be a positive integer")
    path = Path(path)
    chunks = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"video has no audio track: {path}")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not decode audio from {path}: {exc}") from exc
    if not chunks:
        raise ValueError(f"video audio track is empty: {path}")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if audio.size == 0:
        raise ValueError(f"video audio track is empty: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"video audio track contains non-finite samples: {path}")
    return audio


def _normalize_legacy_rope_config(config: Any, transformers_version: str) -> Any:
    """Map the checkpoint's v5 RoPE fields to their equivalent v4.57 representation."""
    try:
        major = int(transformers_version.split(".", 1)[0])
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid Transformers version: {transformers_version!r}") from exc
    if major != 4:
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise ValueError("VA-Judger thinker config has no text_config")
    if getattr(text_config, "rope_scaling", None) is not None:
        return config
    parameters = getattr(text_config, "rope_parameters", None)
    if not isinstance(parameters, Mapping):
        raise ValueError("VA-Judger config lacks compatible RoPE parameters for Transformers 4.x")
    rope_type = parameters.get("rope_type", parameters.get("type"))
    theta = parameters.get("rope_theta")
    sections = parameters.get("mrope_section")
    head_dim = getattr(text_config, "head_dim", None)
    if (
        rope_type != "default"
        or isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("VA-Judger uses unsupported RoPE type or theta for Transformers 4.x")
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in sections)
        or not isinstance(head_dim, int)
        or sum(sections) != head_dim // 2
    ):
        raise ValueError("VA-Judger has invalid multimodal RoPE sections")
    if parameters.get("interleaved") is not True or parameters.get("mrope_interleaved") is not True:
        raise ValueError("VA-Judger RoPE interleaving is incompatible with Transformers 4.x")
    text_config.rope_theta = float(theta)
    text_config.rope_scaling = {
        "type": "default",
        "rope_type": "default",
        "mrope_section": list(sections),
    }
    return config


class NativeVAJudger:
    def __init__(self, args: argparse.Namespace, prompt: str) -> None:
        import torch
        import transformers
        from transformers import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        self.args, self.prompt, self.torch = args, prompt, torch
        self.load_audio = _load_video_audio
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model_path = str(args.reward_model.resolve())
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, max_pixels=args.video_max_pixels, local_files_only=True)
        full_config = Qwen3OmniMoeConfig.from_pretrained(model_path, local_files_only=True)
        model_config = full_config.thinker_config
        _normalize_legacy_rope_config(model_config, transformers.__version__)
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path,
            config=model_config,
            dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation=args.attn_implementation,
            local_files_only=True,
        ).eval()

    def _conversation(self, pair: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Text Caption: {pair['prompt']}\n\nVideo 1:"},
                    {"type": "video", "video": pair["video_1"]},
                    {"type": "text", "text": "Video 2:"},
                    {"type": "video", "video": pair["video_2"]},
                ],
            },
        ]

    def predict(self, pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rendered = self.processor.apply_chat_template(
            [self._conversation(pair) for pair in pairs], add_generation_prompt=True, tokenize=False
        )
        videos = [[pair["video_1"], pair["video_2"]] for pair in pairs]
        video_paths = [path for sample in videos for path in sample]
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio_tracks = [self.load_audio(path, sampling_rate=sampling_rate) for path in video_paths]
        inputs = self.processor(
            text=rendered,
            videos=videos,
            audio=audio_tracks,
            use_audio_in_video=True,
            padding=True,
            return_tensors="pt",
            do_sample_frames=True,
            num_frames=self.args.max_frames,
            max_pixels=self.args.video_max_pixels,
        ).to(device=self.model.device, dtype=self.model.dtype)
        generation: dict[str, Any] = {
            "use_audio_in_video": True,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.do_sample,
            "use_cache": True,
        }
        if self.args.do_sample:
            generation.update(temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, **generation)
        suffix = ids[:, inputs.input_ids.shape[-1] :]
        texts = self.processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(texts) != len(pairs):
            raise VAJudgerError("native generation batch is incomplete")
        return [
            {"id": pair["id"], "dimension_scores": _parse_scores(text), "completion": text}
            for pair, text in zip(pairs, texts, strict=True)
        ]

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite reward artifact: {path}")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def score(args: argparse.Namespace, *, judge_factory: Any = NativeVAJudger) -> Path:
    manifest_path = args.round_manifest.resolve()
    output_path = manifest_path.with_name("rewards.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite reward artifact: {output_path}")
    if args.pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    manifest, raw = _load_manifest(manifest_path)
    if not args.reward_model.resolve().is_dir():
        raise FileNotFoundError(f"reward model directory not found: {args.reward_model}")
    _verify(manifest, True)
    judge = judge_factory(args, SCORING_PROMPT)
    groups = []
    try:
        for group in manifest["groups"]:
            raw_results = []

            def transport(_url: str, payload: Mapping[str, Any], _timeout: float) -> Mapping[str, Any]:
                results = judge.predict(payload["pairs"])
                raw_results.extend(results)
                return {"status": "success", "results": results}

            client = VAJudgerClient("inprocess://native", pair_batch_size=args.pair_batch_size, transport=transport)
            candidates = group["candidates"]
            scores = client.score_group(group["prompt"], [item["path"] for item in candidates])
            groups.append(
                {
                    "group_id": group["group_id"],
                    "prompt": group["prompt"],
                    "candidate_hashes": [item["media_sha256"] for item in candidates],
                    "raw_scores": scores.tolist(),
                    "raw_pair_results": raw_results,
                }
            )
        _verify(manifest, False)
    finally:
        judge.close()
    ignored = {"round_manifest", "reward_model"}
    output = {
        "version": 1,
        "round_id": manifest["round_id"],
        "behavior_sha256": manifest["behavior_sha256"],
        "semantic_fingerprint": manifest["semantic_fingerprint"],
        "start_step": manifest["start_step"],
        "prompt_cursor": manifest["prompt_cursor"],
        "group_count": manifest["group_count"],
        "resume_sha256": manifest["resume_sha256"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "judge": {
            "backend": "native",
            "scoring_prompt_version": SCORING_PROMPT_VERSION,
            "implementation_version": SCORING_IMPLEMENTATION_VERSION,
            "scoring_prompt_sha256": SCORING_PROMPT_SHA256,
            "reward_model": str(args.reward_model.resolve()),
            "args": {k: v for k, v in vars(args).items() if k not in ignored},
        },
        "groups": groups,
    }
    _atomic_write(output_path, output)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    print(score(create_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
