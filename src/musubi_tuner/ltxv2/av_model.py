from __future__ import annotations

from typing import Any, Dict, Optional

from musubi_tuner.ltxv2.model import LTXVModel


class LTXAVModel(LTXVModel):
    def __init__(
        self,
        audio_in_channels: int = 32,
        audio_cross_attention_dim: int = 2048,
        audio_attention_head_dim: int = 64,
        audio_num_attention_heads: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_in_channels = audio_in_channels
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.audio_attention_head_dim = audio_attention_head_dim
        self.audio_num_attention_heads = audio_num_attention_heads

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        frame_rate: int = 25,
        transformer_options: Optional[Dict[str, Any]] = None,
        keyframe_idxs=None,
        **kwargs,
    ):
        raise NotImplementedError(
            "musubi_tuner.ltxv2.av_model.LTXAVModel is a placeholder. Replace with official LTXAV source implementation."
        )
