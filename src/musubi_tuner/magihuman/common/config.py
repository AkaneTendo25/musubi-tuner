# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import sys
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, get_args, get_origin

import torch

from ..utils import env_is_true, print_rank_0


def _coerce_torch_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        if value in ("torch.float32", "float32"):
            return torch.float32
        if value in ("torch.float16", "float16"):
            return torch.float16
        if value in ("torch.bfloat16", "bfloat16"):
            return torch.bfloat16
        if value in ("torch.float8_e4m3fn", "float8_e4m3fn", "fp8_e4m3fn", "fp8"):
            return torch.float8_e4m3fn
        if value in ("torch.float8_e4m3fnuz", "float8_e4m3fnuz", "fp8_e4m3fnuz"):
            return torch.float8_e4m3fnuz
        if value in ("torch.float8_e5m2", "float8_e5m2", "fp8_e5m2"):
            return torch.float8_e5m2
        if value in ("torch.float8_e5m2fnuz", "float8_e5m2fnuz", "fp8_e5m2fnuz"):
            return torch.float8_e5m2fnuz
    raise ValueError(f"Unknown torch.dtype string: '{value}'")


def _strip_saved_config_prefix(text: str) -> str:
    prefix = "MagiPipelineConfig:"
    stripped = text.lstrip()
    if stripped.startswith(prefix):
        return stripped[len(prefix) :].lstrip()
    return text


def _load_json_config(config_load_path: Optional[str]) -> dict[str, Any]:
    if not config_load_path:
        return {}

    text = Path(config_load_path).read_text(encoding="utf-8")
    payload = _strip_saved_config_prefix(text)
    return json.loads(payload)


def _coerce_value(expected_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(expected_type)
    if origin is Union:
        union_args = [arg for arg in get_args(expected_type) if arg is not type(None)]
        if len(union_args) == 1:
            return _coerce_value(union_args[0], value)

    if expected_type is torch.dtype:
        return _coerce_torch_dtype(value)

    if is_dataclass(expected_type):
        return _instantiate_dataclass(expected_type, value)

    if origin in (list, tuple):
        item_types = get_args(expected_type)
        if origin is list:
            item_type = item_types[0] if item_types else Any
            return [_coerce_value(item_type, item) for item in value]
        if origin is tuple:
            if len(item_types) == 2 and item_types[1] is Ellipsis:
                return tuple(_coerce_value(item_types[0], item) for item in value)
            return tuple(_coerce_value(item_type, item) for item_type, item in zip(item_types, value))

    return value


def _instantiate_dataclass(cls: type, data: Optional[dict[str, Any]] = None):
    if data is None:
        data = {}

    kwargs = {}
    for dataclass_field in fields(cls):
        if dataclass_field.name in data:
            kwargs[dataclass_field.name] = _coerce_value(dataclass_field.type, data[dataclass_field.name])
        elif dataclass_field.default is not MISSING:
            kwargs[dataclass_field.name] = copy.deepcopy(dataclass_field.default)
        elif dataclass_field.default_factory is not MISSING:
            kwargs[dataclass_field.name] = dataclass_field.default_factory()
        else:
            raise ValueError(f"Missing required config field: {cls.__name__}.{dataclass_field.name}")

    return cls(**kwargs)


@dataclass
class EngineConfig:
    seed: int = 1234
    load: str | None = None
    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    distributed_timeout_minutes: int = 10
    sequence_parallel: bool = False
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1


@dataclass
class ModelConfig:
    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48 * 4
    audio_in_channels: int = 64
    text_in_channels: int = 3584
    checkpoint_qk_layernorm_rope: bool = False
    params_dtype: torch.dtype | str = torch.float32
    compute_dtype: torch.dtype | str = torch.bfloat16
    tread_config: dict = field(
        default_factory=lambda: dict(selection_rate=0.5, start_layer_idx=2, end_layer_idx=25)
    )
    mm_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 36, 37, 38, 39])
    local_attn_layers: list[int] = field(default_factory=list)
    enable_attn_gating: bool = True
    activation_type: str = "swiglu7"
    gelu7_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    num_heads_q: int = 0
    num_heads_kv: int = 0
    post_norm_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.params_dtype = _coerce_torch_dtype(self.params_dtype)
        self.compute_dtype = _coerce_torch_dtype(self.compute_dtype)


@dataclass
class DataProxyConfig:
    t_patch_size: int = 1
    patch_size: int = 2
    frame_receptive_field: int = 11
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    ref_audio_offset: int = 1000
    text_offset: int = 0
    coords_style: Literal["v1", "v2"] = "v2"


@dataclass
class EvaluationConfig:
    data_proxy_config: DataProxyConfig = field(default_factory=DataProxyConfig)
    fps: int = 25
    num_inference_steps: int = 32
    video_txt_guidance_scale: float = 5.0
    audio_txt_guidance_scale: float = 5.0
    txt_encoder_type: Literal["t5_gemma"] = "t5_gemma"
    t5_gemma_target_length: int = 640
    support_ref_audio: bool = True
    shift: float = 5.0
    exp_name: str = "exp_debug"
    audio_model_path: str = ""
    txt_model_path: str = ""
    vae_model_path: str = ""
    vae_stride: Tuple[int, int, int] = (4, 16, 16)
    z_dim: int = 48
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    cfg_number: int = 2
    sr_cfg_number: int = 2
    enable_flops_recording: bool = False
    use_sr_model: bool = False
    sr_model_path: str = ""
    sr_num_inference_steps: int = 5
    noise_value: int = 220
    sr_video_txt_guidance_scale: float = 3.5
    use_cfg_trick: bool = True
    cfg_trick_start_frame: int = 13
    cfg_trick_value: float = 2.0
    using_sde_flag: bool = False
    sr_audio_noise_scale: float = 0.7
    use_turbo_vae: bool = True
    student_config_path: str = ""
    student_ckpt_path: str = ""


@dataclass
class MagiPipelineConfig:
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    arch_config: ModelConfig = field(default_factory=ModelConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    sr_arch_config: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        self.validate_engine_config()
        self.post_override_config()

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        data = asdict(self)
        data["arch_config"]["params_dtype"] = str(self.arch_config.params_dtype)
        data["arch_config"]["compute_dtype"] = str(self.arch_config.compute_dtype)
        data["sr_arch_config"]["params_dtype"] = str(self.sr_arch_config.params_dtype)
        data["sr_arch_config"]["compute_dtype"] = str(self.sr_arch_config.compute_dtype)
        return data

    def save_to_json(self, json_path: str, indent: int = 4):
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False), encoding="utf-8")

    def __str__(self, indent: int = 4):
        formatted = json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False, sort_keys=False)
        return f"MagiPipelineConfig:\n{formatted}"

    def __repr__(self):
        return self.__str__()

    def validate_engine_config(self):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.engine_config.dp_size = world_size // (
            self.engine_config.tp_size * self.engine_config.pp_size * self.engine_config.cp_size
        )

        assert world_size % self.engine_config.tp_size == 0
        tp_pp_size = self.engine_config.tp_size * self.engine_config.pp_size
        assert world_size % tp_pp_size == 0
        tp_pp_cp_size = tp_pp_size * self.engine_config.cp_size
        assert world_size % tp_pp_cp_size == 0
        assert world_size == self.engine_config.dp_size * tp_pp_cp_size

        if self.engine_config.tp_size == 1:
            self.engine_config.sequence_parallel = False

        return self

    def post_override_config(self):
        self.arch_config.num_heads_q = self.arch_config.hidden_size // self.arch_config.head_dim
        self.arch_config.num_heads_kv = self.arch_config.num_query_groups

        self.sr_arch_config = copy.deepcopy(self.arch_config)
        if env_is_true("SR2_1080"):
            self.sr_arch_config = copy.deepcopy(self.arch_config)
            self.sr_arch_config.local_attn_layers = [
                0, 1, 2,
                4, 5, 6,
                8, 9, 10,
                12, 13, 14,
                16, 17, 18,
                20, 21, 22,
                24, 25, 26,
                28, 29, 30,
                32, 33, 34,
                35, 36, 37,
                38, 39,
            ]
            self.evaluation_config.sr_video_txt_guidance_scale = 3.5

        return self


def prevent_unsupported_list_syntax():
    """
    Keep the old error message behavior for unsupported list CLI syntax.
    """
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if index + 2 < len(args):
            value1, value2 = args[index + 1], args[index + 2]
            if not value1.startswith("-") and not value2.startswith("-"):
                error_msg = (
                    f"\n\nError: Detected list parameter '{arg}' using unsupported command line syntax.\n"
                    f"Error pattern: '{arg} {value1} {value2} ...'\n\n"
                    "Use one of the following supported formats:\n\n"
                    f"1. JSON style:      {arg} '[{value1},{value2},...]'\n"
                    f"2. Argparse style:  {arg} {value1} {arg} {value2}\n"
                    f"3. Lazy style:      {arg} {value1},{value2}\n"
                )
                raise ValueError(error_msg)


def parse_config(verbose: bool = False) -> MagiPipelineConfig:
    parser = argparse.ArgumentParser(description="Load and optionally save config", allow_abbrev=False)
    parser.add_argument("--config-load-path", type=str, default=None, help="Path to load the config JSON from")
    parser.add_argument("--config-save-path", type=str, default=None, help="Path to save the config JSON to")
    args, _ = parser.parse_known_args()

    prevent_unsupported_list_syntax()

    raw_config = _load_json_config(args.config_load_path)
    config = _instantiate_dataclass(MagiPipelineConfig, raw_config)

    if args.config_save_path is not None:
        config.save_to_json(args.config_save_path)

    if verbose:
        print_rank_0(config)

    return config
