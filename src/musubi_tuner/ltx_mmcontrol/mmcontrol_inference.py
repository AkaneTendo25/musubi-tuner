"""MMControl inference helpers for LTX-2 bypass modules."""

from __future__ import annotations

from musubi_tuner.ltx_vace.vace_inference import AudioVaceInferenceHelper, VaceInferenceHelper

MMCONTROL_DEFAULT_LAYERS = tuple(range(0, 48, 2))


class MMControlInferenceHelper(VaceInferenceHelper):
    """Visual MMControl helper using even-layer defaults."""

    def __init__(
        self,
        mmcontrol_model_path: str,
        mmcontrol_layers=MMCONTROL_DEFAULT_LAYERS,
        mmcontrol_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            vace_model_path=mmcontrol_model_path,
            vace_layers=mmcontrol_layers,
            vace_scale=mmcontrol_scale,
            **kwargs,
        )


class AudioMMControlInferenceHelper(AudioVaceInferenceHelper):
    """Audio MMControl helper using even-layer defaults."""

    def __init__(
        self,
        audio_mmcontrol_model_path: str,
        mmcontrol_layers=MMCONTROL_DEFAULT_LAYERS,
        mmcontrol_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            audio_vace_model_path=audio_mmcontrol_model_path,
            vace_layers=mmcontrol_layers,
            vace_scale=mmcontrol_scale,
            **kwargs,
        )