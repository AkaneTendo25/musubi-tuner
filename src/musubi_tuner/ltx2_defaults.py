"""LTX-2 sampling presets shared by training and standalone generation."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


@dataclass(frozen=True)
class LTX2SamplingPreset:
    sample_steps: int
    width: int
    height: int
    frame_count: int
    frame_rate: float
    video_cfg_scale: float
    audio_cfg_scale: float
    stg_scale: float
    stg_blocks: list[int]
    stg_mode: str
    video_rescale_scale: float
    audio_rescale_scale: float
    video_modality_scale: float
    audio_modality_scale: float
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


LTX2_SAMPLING_PRESETS: dict[str, LTX2SamplingPreset] = {
    "ltx20": LTX2SamplingPreset(
        sample_steps=40,
        width=768,
        height=512,
        frame_count=121,
        frame_rate=24.0,
        video_cfg_scale=3.0,
        audio_cfg_scale=7.0,
        stg_scale=1.0,
        stg_blocks=[29],
        stg_mode="both",
        video_rescale_scale=0.7,
        audio_rescale_scale=0.7,
        video_modality_scale=3.0,
        audio_modality_scale=3.0,
    ),
    "ltx23": LTX2SamplingPreset(
        sample_steps=15,
        width=960,
        height=544,
        frame_count=121,
        frame_rate=24.0,
        video_cfg_scale=3.0,
        audio_cfg_scale=7.0,
        stg_scale=1.0,
        stg_blocks=[28],
        stg_mode="both",
        video_rescale_scale=0.9,
        audio_rescale_scale=0.9,
        video_modality_scale=3.0,
        audio_modality_scale=3.0,
    ),
    "ltx23_hq": LTX2SamplingPreset(
        sample_steps=15,
        width=1920,
        height=1088,
        frame_count=121,
        frame_rate=24.0,
        video_cfg_scale=3.0,
        audio_cfg_scale=7.0,
        stg_scale=0.0,
        stg_blocks=[],
        stg_mode="both",
        video_rescale_scale=0.45,
        audio_rescale_scale=1.0,
        video_modality_scale=3.0,
        audio_modality_scale=3.0,
    ),
    # Distilled two-stage generation defaults. Callers still need the distilled
    # checkpoint/LoRA and spatial upsampler for the refinement stage.
    "distilled_two_stage": LTX2SamplingPreset(
        sample_steps=8,
        width=960,
        height=544,
        frame_count=121,
        frame_rate=24.0,
        video_cfg_scale=1.0,
        audio_cfg_scale=1.0,
        stg_scale=0.0,
        stg_blocks=[],
        stg_mode="both",
        video_rescale_scale=0.0,
        audio_rescale_scale=0.0,
        video_modality_scale=1.0,
        audio_modality_scale=1.0,
        negative_prompt="",
    ),
}


def get_ltx2_sampling_preset(name: str | None, *, ltx_version: str = "2.3") -> LTX2SamplingPreset | None:
    if not name or name == "legacy":
        return None
    if name == "defaults":
        name = "ltx23" if str(ltx_version) == "2.3" else "ltx20"
    try:
        return LTX2_SAMPLING_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(["legacy", "defaults", *LTX2_SAMPLING_PRESETS.keys()])
        raise ValueError(f"Unknown LTX-2 sampling preset '{name}'. Expected one of: {valid}") from exc
