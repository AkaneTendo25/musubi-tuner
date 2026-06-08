"""Inference-only VideoScore2 scorer.

Implements the same scoring pipeline as the TIGER-Lab/VideoScore2 model card
(VideoScore2Worker + the soft-score / token-index / parse helpers) as an IN-PROCESS
scorer: no fastapi / uvicorn / multi-GPU worker pool, just the model load + scoring math.

VideoScore2 is a stock Qwen2_5_VLForConditionalGeneration checkpoint (4 safetensors
shards, no custom modeling, no reward head, no LoRA), loaded via AutoModelForVision2Seq /
AutoProcessor. The scorer builds the VS2 query
prompt, builds the Qwen2.5-VL chat ({type:video, video, fps} + the query text), runs
process_vision_info (prefers the installed qwen_vl_utils -- exactly what the model card
uses -- and falls back to the self-contained vendored copy), runs model.generate with
output_scores=True / return_dict_in_generate=True, parses the three hard scores out of the
generated text, locates the score-digit token index for each dimension, and reads a soft
score off log_softmax over the digit tokens "1".."5":
round(best_score * (max_prob / total_prob), 4).

Runtime notes:
  * No HTTP server / worker pool / cv2 path validation: score() is a direct call.
  * AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True) is kept; the
    released checkpoint is a stock transformers architecture, so it loads natively, but the
    flag is harmless.
  * attn_implementation follows the same flash_attn->sdpa policy as the vendored hpsv3 /
    videoreward copies: pin flash_attention_2 when flash_attn is importable, else sdpa, for
    parity with the sibling rewards on the same box.
  * Generation defaults: do_sample=True, temperature=0.7, max_new_tokens=1024; a seed arg is
    added so a reward can be reproducible, and do_sample can be set False for
    greedy/deterministic scoring.

Public API: VideoScore2Inferencer (.score(video_path, prompt, ...) -> dict).
"""

from __future__ import annotations

import logging
import re
from string import Template

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)

# Verbatim from the VS2_QUERY_TEMPLATE on the model card. The
# three en-dash (U+2013) characters are reproduced via – escapes so this
# vendored file stays pure-ASCII on disk while the runtime string is byte-identical to the
# upstream (the prompt text drives tokenization, so it must match exactly).
_DASH = "–"
VS2_QUERY_TEMPLATE = Template(
    "\n"
    "You are an expert for evaluating AI-generated videos from three dimensions:\n"
    "(1) visual quality " + _DASH + " clarity, smoothness, artifacts;\n"
    "(2) text-to-video alignment " + _DASH + " fidelity to the prompt;\n"
    "(3) physical/common-sense consistency " + _DASH + " naturalness and physics plausibility.\n"
    "\n"
    "Video prompt: $t2v_prompt\n"
    "\n"
    "Please output in this format:\n"
    "visual quality: <v_score>;\n"
    "text-to-video alignment: <t_score>,\n"
    "physical/common-sense consistency: <p_score>\n"
)

# Generation defaults from the model card.
DEFAULT_INFER_FPS = 2.0
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7


def _process_vision_info(messages):
    """Prefer the installed qwen_vl_utils.process_vision_info (exactly what the model card
    uses, so byte-identical preprocessing), else the self-contained vendored fallback.
    Returns (image_inputs, video_inputs) -- the 2-value form the upstream calls.
    """
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except ImportError:
        from .vision_process import process_vision_info
    return process_vision_info(messages)


def _ll_based_soft_score_normed(hard_val, token_idx, scores, tokenizer):
    """Verbatim from the model card's _ll_based_soft_score_normed.

    Soft score = best integer score (1..5) weighted by its normalized probability read off
    the generation logits at the score-digit token position.
    """
    if hard_val is None or token_idx < 0:
        return None
    logits = scores[token_idx][0]
    score_probs = []
    for s in range(1, 6):
        ids = tokenizer.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            logp = torch.log_softmax(logits, dim=-1)[ids[0]].item()
            score_probs.append((s, float(np.exp(logp))))
    if not score_probs:
        return None
    scores_list, probs_list = zip(*score_probs)
    total_prob = sum(probs_list)
    max_prob = max(probs_list)
    best_score = scores_list[probs_list.index(max_prob)]
    normalized_prob = max_prob / total_prob if total_prob > 0 else 0
    return round(best_score * normalized_prob, 4)


def _find_score_token_index_by_prompt(prompt_text, tokenizer, gen_ids):
    """Verbatim from the model card's _find_score_token_index_by_prompt.

    Finds the generated-token index of the first digit that follows prompt_text
    (e.g. "visual quality:") so the soft score is read from the right logits row.
    """
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
    match = re.search(pattern, gen_str, flags=re.IGNORECASE)
    if not match:
        return -1
    after_text = gen_str[match.end() :]
    num_match = re.search(r"\d", after_text)
    if not num_match:
        return -1
    target_substr = gen_str[: match.end() + num_match.start() + 1]
    for i in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False)
        if partial == target_substr:
            return i
    return -1


_VQ_LABEL = "visual quality:"
_TA_LABEL = "text-to-video alignment:"
_PC_LABEL = "physical/common-sense consistency:"

# Tolerant per-dimension phrase patterns (no trailing colon). Some VideoScore2 checkpoints
# do NOT emit the source's "visual quality: N" format; they produce a <think>...</think>
# chain-of-thought then echo the numbered dimension list with the score after the
# description, e.g. "(1) visual quality - clarity, smoothness, artifacts: 3". The phrase +
# "[^\n:]*:\s*(\d)" form matches BOTH that and the strict source format (where [^\n:]* is
# empty), so it is a superset fallback used only when the strict source regex misses.
_VQ_PHRASE = r"visual quality"
_TA_PHRASE = r"text[\s-]*to[\s-]*video alignment"
_PC_PHRASE = r"physical/common[\s-]*sense consistency"


def _final_answer_region(text):
    """The score block follows the optional CoT; read it from after the last </think> so
    think-section prose ("the visual quality is moderate (score 3)") can't shadow the real
    list. Falls back to the whole text when there is no </think> (e.g. strict source format).
    """
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def _hard_from_phrase(phrase_regex, text):
    """First 1..5 digit that follows `<phrase> ... :` on a single line (tolerant fallback)."""
    m = re.search(phrase_regex + r"[^\n:]*:\s*([1-5])", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_score_token_index_by_phrase(phrase_regex, tokenizer, gen_ids):
    """Tolerant-format counterpart of _find_score_token_index_by_prompt: locate the
    generated-token index of the score digit that follows `<phrase> ... :`, preferring the
    final answer region so the soft score is read off the right logits row.
    """
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pat = phrase_regex + r"[^\n:]*:\s*([1-5])"
    cut = gen_str.rfind("</think>")
    base = cut if cut != -1 else 0
    match = re.search(pat, gen_str[base:], flags=re.IGNORECASE)
    if not match:
        base, match = 0, re.search(pat, gen_str, flags=re.IGNORECASE)
        if not match:
            return -1
    digit_end = base + match.end()  # index just past the captured digit
    target_substr = gen_str[:digit_end]
    for i in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False)
        if partial == target_substr:
            return i
    return -1


def _build_hard_score_pattern():
    """Assemble the _parse_scores hard-score regex from the label constants."""
    s = r"\s*"
    d = r"(\d+)"
    gap = r".*?"
    return _VQ_LABEL + s + d + gap + _TA_LABEL + s + d + gap + _PC_LABEL + s + d


def _parse_scores(output_text, gen_token_ids, scores, tokenizer):
    """_parse_scores: hard scores via regex over the generated text
    + soft scores via the per-dimension digit logits. Returns the upstream dict shape.
    """
    match = re.search(_build_hard_score_pattern(), output_text, re.DOTALL | re.IGNORECASE)
    if match:
        v_hard = int(match.group(1))
        t_hard = int(match.group(2))
        p_hard = int(match.group(3))
    else:
        # Strict source format absent -> tolerant per-dimension parse over the final answer
        # region (handles the "(1) visual quality - ...: N" numbered form this checkpoint
        # emits). Retry over the whole text if the region yielded nothing (e.g. truncated CoT).
        region = _final_answer_region(output_text)
        v_hard = _hard_from_phrase(_VQ_PHRASE, region)
        t_hard = _hard_from_phrase(_TA_PHRASE, region)
        p_hard = _hard_from_phrase(_PC_PHRASE, region)
        if v_hard is None and t_hard is None and p_hard is None:
            v_hard = _hard_from_phrase(_VQ_PHRASE, output_text)
            t_hard = _hard_from_phrase(_TA_PHRASE, output_text)
            p_hard = _hard_from_phrase(_PC_PHRASE, output_text)

    idx_v = _find_score_token_index_by_prompt(_VQ_LABEL, tokenizer, gen_token_ids)
    idx_t = _find_score_token_index_by_prompt(_TA_LABEL, tokenizer, gen_token_ids)
    idx_p = _find_score_token_index_by_prompt(_PC_LABEL, tokenizer, gen_token_ids)
    if idx_v < 0:
        idx_v = _find_score_token_index_by_phrase(_VQ_PHRASE, tokenizer, gen_token_ids)
    if idx_t < 0:
        idx_t = _find_score_token_index_by_phrase(_TA_PHRASE, tokenizer, gen_token_ids)
    if idx_p < 0:
        idx_p = _find_score_token_index_by_phrase(_PC_PHRASE, tokenizer, gen_token_ids)

    v_soft = _ll_based_soft_score_normed(v_hard, idx_v, scores, tokenizer)
    t_soft = _ll_based_soft_score_normed(t_hard, idx_t, scores, tokenizer)
    p_soft = _ll_based_soft_score_normed(p_hard, idx_p, scores, tokenizer)
    out = {}
    out["visual_quality"] = v_soft
    out["text_to_video_alignment"] = t_soft
    out["physical_consistency"] = p_soft
    out["visual_quality_hard"] = v_hard
    out["text_to_video_alignment_hard"] = t_hard
    out["physical_consistency_hard"] = p_hard
    out["raw_output"] = output_text
    return out


class VideoScore2Inferencer:
    """In-process VideoScore2 scorer.

    model_name_or_path is the local VideoScore2 checkpoint dir (or HF id). device is e.g.
    "cuda" / "cuda:1" / "cpu". The model is loaded once; score() runs the full generate +
    parse pipeline for one video and returns the score dict.
    """

    def __init__(self, model_name_or_path, device="cuda", dtype=torch.bfloat16):
        self.device = device

        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=False
        )
        logger.info("VideoScore2: model loaded on %s (attn=%s)", device, attn_impl)

    @torch.no_grad()
    def score(
        self,
        video_path,
        prompt,
        fps=DEFAULT_INFER_FPS,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        do_sample=True,
        seed=None,
    ):
        """Score one video -> the parsed score dict."""
        if seed is not None:
            torch.manual_seed(int(seed))
        user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=prompt)
        vid_item = dict(type="video", video=video_path, fps=fps)
        txt_item = dict(type="text", text=user_prompt)
        messages = [dict(role="user", content=[vid_item, txt_item])]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        gen_out = self.model.generate(**inputs, **gen_kwargs)
        sequences = gen_out.sequences
        scores = gen_out.scores
        input_len = inputs["input_ids"].shape[1]
        gen_token_ids = sequences[0, input_len:].tolist()
        output_text = self.processor.batch_decode(
            sequences[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _parse_scores(output_text, gen_token_ids, scores, self.tokenizer)
