from typing import Optional, Tuple

import torch
from musubi_tuner.ltx_2.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings

ADALN_BASE_PARAMS_COUNT = 6
ADALN_CROSS_ATTN_PARAMS_COUNT = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    return ADALN_BASE_PARAMS_COUNT + (ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else 0)


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
