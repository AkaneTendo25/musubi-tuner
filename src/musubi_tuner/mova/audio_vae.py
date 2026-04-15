import importlib
import importlib.util
import os
from types import ModuleType
from typing import Any, Optional

import torch


def _load_module_from_file(path: str) -> ModuleType:
    module_name = os.path.splitext(os.path.basename(path))[0] + "_mova_import"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_symbol(spec: str, default_candidates: Optional[list[str]] = None) -> Any:
    if ":" in spec:
        module_spec, symbol_name = spec.split(":", 1)
    else:
        module_spec, symbol_name = spec, None

    if os.path.isfile(module_spec):
        module = _load_module_from_file(module_spec)
    else:
        module = importlib.import_module(module_spec)

    if symbol_name is not None:
        return getattr(module, symbol_name)

    default_candidates = default_candidates or []
    for candidate in default_candidates:
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise ImportError(f"Could not resolve a symbol from '{spec}'")


def _extract_latents(encode_output: Any) -> torch.Tensor:
    if isinstance(encode_output, torch.Tensor):
        latents = encode_output
    elif isinstance(encode_output, (tuple, list)):
        tensor_like = next((item for item in encode_output if isinstance(item, torch.Tensor)), None)
        if tensor_like is None:
            raise ValueError(f"Unsupported audio VAE encode output: {type(encode_output)}")
        latents = tensor_like
    elif hasattr(encode_output, "latent_dist"):
        latent_dist = encode_output.latent_dist
        if hasattr(latent_dist, "sample"):
            latents = latent_dist.sample()
        elif hasattr(latent_dist, "mode"):
            latents = latent_dist.mode()
        else:
            raise ValueError("audio VAE latent_dist has no sample/mode method")
    elif hasattr(encode_output, "latents"):
        latents = encode_output.latents
    else:
        raise ValueError(f"Unsupported audio VAE encode output: {type(encode_output)}")

    if latents.ndim == 4 and latents.shape[2] == 1:
        latents = latents.squeeze(2)
    if latents.ndim != 3:
        raise ValueError(f"Expected audio latents [B, C, L], got {tuple(latents.shape)}")
    return latents


def _extract_waveform(decode_output: Any) -> torch.Tensor:
    if isinstance(decode_output, torch.Tensor):
        waveform = decode_output
    elif isinstance(decode_output, (tuple, list)):
        tensor_like = next((item for item in decode_output if isinstance(item, torch.Tensor)), None)
        if tensor_like is None:
            raise ValueError(f"Unsupported audio VAE decode output: {type(decode_output)}")
        waveform = tensor_like
    elif hasattr(decode_output, "sample"):
        waveform = decode_output.sample
    elif hasattr(decode_output, "waveform"):
        waveform = decode_output.waveform
    elif hasattr(decode_output, "audio_values"):
        waveform = decode_output.audio_values
    elif hasattr(decode_output, "audios"):
        waveform = decode_output.audios
    else:
        raise ValueError(f"Unsupported audio VAE decode output: {type(decode_output)}")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected decoded waveform [B, C, T], got {tuple(waveform.shape)}")
    return waveform


class MovaAudioVAEAdapter:
    def __init__(self, model: torch.nn.Module, device: torch.device, dtype: torch.dtype):
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.sample_rate = (
            getattr(model, "sample_rate", None)
            or getattr(getattr(model, "config", None), "sample_rate", None)
            or getattr(getattr(model, "config", None), "sampling_rate", None)
            or 44100
        )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            latents = _extract_latents(self.model.encode(waveform))
        return latents.to(dtype=self.dtype)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            waveform = _extract_waveform(self.model.decode(latents))
        return waveform.to(dtype=torch.float32)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        self.model = self.model.to(device=device, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        return self

    def eval(self):
        self.model.eval()
        return self

    def requires_grad_(self, requires_grad: bool = False):
        self.model.requires_grad_(requires_grad)
        return self


def load_audio_vae(
    pretrained_model_path: str,
    *,
    subfolder: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    vae_type: str = "dac",
    model_spec: Optional[str] = None,
) -> MovaAudioVAEAdapter:
    if vae_type == "oobleck":
        from diffusers import AutoencoderOobleck

        model_cls = AutoencoderOobleck
    else:
        candidate_specs = []
        if model_spec is not None:
            candidate_specs.append(model_spec)
        candidate_specs.extend(
            [
                "dac:DAC",
                "dac.model:DAC",
                "descript_audio_codec.dac.model.dac:DAC",
            ]
        )

        model_cls = None
        for candidate_spec in candidate_specs:
            try:
                model_cls = import_symbol(candidate_spec, ["DAC"])
                break
            except Exception:
                continue
        if model_cls is None:
            raise ImportError(
                "Could not import a DAC audio VAE class. "
                "Pass --audio_vae_model_spec with a module path like "
                "'path/to/dac_vae.py:DAC' or install a compatible DAC package."
            )

    kwargs = {"torch_dtype": dtype}
    if subfolder:
        kwargs["subfolder"] = subfolder

    if hasattr(model_cls, "from_pretrained"):
        model = model_cls.from_pretrained(pretrained_model_path, **kwargs)
    else:
        target_path = os.path.join(pretrained_model_path, subfolder) if subfolder else pretrained_model_path
        model = model_cls(target_path)

    return MovaAudioVAEAdapter(model, device=device, dtype=dtype)
