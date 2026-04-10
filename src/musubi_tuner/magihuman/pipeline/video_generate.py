from dataclasses import dataclass

import torch


@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor
