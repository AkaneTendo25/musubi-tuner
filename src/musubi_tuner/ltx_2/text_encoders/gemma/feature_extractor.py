import math

import torch
from einops import rearrange

from musubi_tuner.ltx_2.model.model_protocol import ModelConfigurator


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")
    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    return normed.masked_fill(~mask_flattened, 0.0)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    b, _t, d, l = encoded_text.shape  # noqa: E741
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B,T,1,L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(b, attention_mask.shape[1], d * l)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


class GemmaFeaturesExtractorProjDualLinear(torch.nn.Module):
    def __init__(
        self,
        *,
        video_out_dim: int,
        audio_out_dim: int | None = None,
        in_dim: int = 3840 * 49,
    ) -> None:
        super().__init__()
        self.embedding_dim = 3840
        self.video_aggregate_embed = torch.nn.Linear(in_dim, video_out_dim, bias=True)
        self.audio_aggregate_embed = (
            torch.nn.Linear(in_dim, audio_out_dim, bias=True) if audio_out_dim is not None else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = _norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)

        video_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(normed * math.sqrt(video_dim / self.embedding_dim))

        audio = None
        if self.audio_aggregate_embed is not None:
            audio_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(normed * math.sqrt(audio_dim / self.embedding_dim))
        return video, audio


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    _V2_EXPECTED_CONFIG = {
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_proj_input_norm": False,
        "caption_projection_second_linear": False,
    }

    def __init__(self, is_av: bool = True) -> None:
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)
        self.is_av = is_av

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side)
        features = self.aggregate_embed(normed.to(encoded.dtype))
        return (features, features) if self.is_av else (features, None)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], config: dict) -> torch.nn.Module:
        transformer = config.get("transformer", {})
        overlap = set(transformer.keys()) & set(cls._V2_EXPECTED_CONFIG.keys())
        if not overlap:
            return cls(is_av=True)

        if transformer.get("caption_proj_before_connector", False):
            video_out_dim = transformer.get("num_attention_heads", 32) * transformer.get("attention_head_dim", 128)
            audio_out_dim = transformer.get("audio_num_attention_heads", 32) * transformer.get(
                "audio_attention_head_dim", 64
            )
            return GemmaFeaturesExtractorProjDualLinear(video_out_dim=video_out_dim, audio_out_dim=audio_out_dim)

        return cls(is_av=True)
