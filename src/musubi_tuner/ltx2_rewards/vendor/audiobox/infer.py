# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Vendored inference-only copy of ``audiobox_aesthetics.infer`` (v0.0.4), with the
# training / CLI / download surface stripped:
#   - ``tqdm`` removed (only used by the CLI ``main_predict``).
#   - ``json`` / ``re`` retained where the inference path needs them (``re`` strips
#     the ``model.`` state_dict prefix exactly as upstream; ``json`` is unused now
#     that ``load_dataset`` / ``main_predict`` are dropped, so it too is removed).
#   - ``from .utils import load_model`` removed: that helper only auto-downloaded the
#     checkpoint from HF / S3. The reward plugin always passes a local checkpoint, so
#     we inline a plain local-path check and raise a clear error otherwise.
#   - ``AesMultiOutput.from_pretrained`` HF fallback dropped (would pull huggingface_hub).
#   - ``load_dataset`` / ``main_predict`` CLI helpers dropped.
# The scoring path (``read_wav``, ``make_inference_batch``, ``AesPredictor``,
# ``initialize_predictor``) is byte-for-byte identical to upstream.

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
import torch.nn.functional as F

from .model.aes import AesMultiOutput, Normalize

# Create module-level logger instead of configuring root logger
logger = logging.getLogger(__name__)


# STRUCT
Batch = Dict[str, Any]

# CONST
AXES_NAME = ["CE", "CU", "PC", "PQ"]


def read_wav(meta):
    path = meta["path"]

    if "start_time" in meta:
        start = meta["start_time"]
        end = meta["end_time"]
        sr = torchaudio.info(path).sample_rate
        wav, _ = torchaudio.load(path, frame_offset=start * sr, num_frames=(end - start) * sr)
    else:
        wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    return wav, sr


def make_inference_batch(
    input_wavs: list,
    hop_size=10,
    window_size=10,
    sample_rate=16000,
    pad_zero=True,
):
    wavs = []
    masks = []
    weights = []
    bids = []
    offset = hop_size * sample_rate
    winlen = window_size * sample_rate
    for bid, wav in enumerate(input_wavs):
        for ii in range(0, wav.shape[-1], offset):
            wav_ii = wav[..., ii : ii + winlen]
            wav_ii_len = wav_ii.shape[-1]
            if wav_ii_len < winlen and pad_zero:
                wav_ii = F.pad(wav_ii, (0, winlen - wav_ii_len))
            mask_ii = torch.zeros_like(wav_ii, dtype=torch.bool)
            mask_ii[:, 0:wav_ii_len] = True
            wavs.append(wav_ii)
            masks.append(mask_ii)
            weights.append(wav_ii_len / winlen)
            bids.append(bid)
    return wavs, masks, weights, bids


@dataclass
class AesPredictor:
    checkpoint_pth: str
    precision: str = "bf16"
    batch_size: int = 1
    data_col: str = "path"
    sample_rate: int = 16000  # const

    def __post_init__(self):
        self.setup_model()

    def setup_model(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Setting up Aesthetic model on {self.device}")

        if self.checkpoint_pth is not None:
            logger.info("Using local checkpoint ...")
            # Original way to load model directly using load_state_dict
            checkpoint_file = self.checkpoint_pth
            if not Path(checkpoint_file).exists():
                raise FileNotFoundError(
                    f"audiobox checkpoint not found: {checkpoint_file}. "
                    "Pass a local Lightning-style checkpoint.pt (state_dict + "
                    "model_cfg + target_transform)."
                )

            # Rename keys
            with open(checkpoint_file, "rb") as fin:
                ckpt = torch.load(fin, map_location=self.device)
                state_dict = {re.sub("^model.", "", k): v for (k, v) in ckpt["state_dict"].items()}

            model = AesMultiOutput(
                **(
                    {
                        k: ckpt["model_cfg"][k]
                        for k in [
                            "proj_num_layer",
                            "proj_ln",
                            "proj_act_fn",
                            "proj_dropout",
                            "nth_layer",
                            "use_weighted_layer_sum",
                            "precision",
                            "normalize_embed",
                            "output_dim",
                        ]
                    }
                    | {"target_transform": ckpt["target_transform"]}
                )
            )
            model.load_state_dict(state_dict)
        else:
            raise ValueError(
                "audiobox vendored predictor requires a local checkpoint_pth; the "
                "HF from_pretrained fallback is not bundled (it would pull "
                "huggingface_hub). Pass checkpoint_path=<checkpoint.pt>."
            )

        model.to(self.device)
        model.eval()

        self.model = model

        self.target_transform = {
            axis: Normalize(
                mean=model.target_transform[axis]["mean"],
                std=model.target_transform[axis]["std"],
            )
            for axis in AXES_NAME
        }

    def audio_resample_mono(self, data_list: List[Batch]) -> List:
        wavs = []
        for ii, item in enumerate(data_list):
            if isinstance(item[self.data_col], str):
                # wav, sr = torchaudio.load(item[self.data_col])
                wav, sr = read_wav(item)
            else:
                wav = item[self.data_col]
                sr = item["sample_rate"]

            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )
            wav = wav.mean(dim=0, keepdim=True)
            wavs.append(wav)
        return wavs

    def forward(self, batch):
        with torch.inference_mode():
            bsz = len(batch)
            wavs = self.audio_resample_mono(batch)
            wavs, masks, weights, bids = make_inference_batch(
                wavs,
                10,
                10,
                sample_rate=self.sample_rate,
            )

            # collate
            wavs = torch.stack(wavs).to(self.device)
            masks = torch.stack(masks).to(self.device)
            weights = torch.tensor(weights).to(self.device)
            bids = torch.tensor(bids).to(self.device)

            assert wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0]
            preds_all = self.model({"wav": wavs, "mask": masks})
            all_result = {}
            for axis in AXES_NAME:
                preds = self.target_transform[axis].inverse(preds_all[axis])
                weighted_preds = []
                for bii in range(bsz):
                    weights_bii = weights[bids == bii]
                    weighted_preds.append(((preds[bids == bii] * weights_bii).sum() / weights_bii.sum()).item())
                all_result[axis] = weighted_preds
            # re-arrenge result
            all_rows = [dict(zip(all_result.keys(), vv)) for vv in zip(*all_result.values())]
            return all_rows


def initialize_predictor(ckpt=None):
    model_predictor = AesPredictor(checkpoint_pth=ckpt, data_col="path")
    return model_predictor
