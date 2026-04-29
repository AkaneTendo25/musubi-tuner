"""LTX-2 sampling and inference methods (mixin for LTX2NetworkTrainer)."""

import argparse
import copy
import gc
import logging
import locale
import os
import subprocess
import sys
import tempfile
import time
import wave
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from accelerate import Accelerator, PartialState

from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils import model_utils
from musubi_tuner.hv_train_network import load_prompts, should_sample_images
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.ltx2_inference import LTX2Inferencer, InferenceConfig
from musubi_tuner.ltx2_defaults import get_ltx2_sampling_preset
from musubi_tuner.ltx2_lycoris_runtime import (
    ensure_adapters_enabled_for_sampling,
    get_adapter_norm_samples,
    summarize_active_adapters,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_PROMPTS_CACHE = "ltx2_sample_prompts_cache.pt"
DEFAULT_SAMPLE_LATENTS_CACHE = "ltx2_sample_latents_cache.pt"


def _decode_subprocess_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp1251", "cp866"):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def infer_ic_lora_strategy_from_preset(lora_target_preset: Optional[str]) -> str:
    """Infer IC-LoRA strategy from LoRA target preset for backward-compatible auto mode."""
    preset = str(lora_target_preset or "").lower()
    if preset == "v2v":
        return "v2v"
    if preset == "audio_ref_only_ic":
        return "audio_ref_only_ic"
    if preset == "av_ic":
        return "av_ic"
    if preset == "video_ref_only_av":
        return "video_ref_only_av"
    return "none"


def _normalize_av_cross_attention_mode(value: Optional[str]) -> str:
    mode = str(value or "both").lower()
    if mode not in {"both", "a2v_only", "v2a_only", "none"}:
        raise ValueError(
            "av_cross_attention_mode must be one of ['both', 'a2v_only', 'v2a_only', 'none']"
        )
    return mode


class LTX2SamplingMixin:

    def _get_audio_preview_config(self, args: argparse.Namespace, transformer) -> Dict[str, int | float]:
        if self._audio_preview_config is not None:
            return self._audio_preview_config

        from musubi_tuner.ltx_2.model.audio_vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR

        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for audio preview config")

        config = self._load_ltx2_checkpoint_config(args)
        audio_vae_cfg = config.get("audio_vae", {})
        model_cfg = audio_vae_cfg.get("model", {}).get("params", {})
        ddconfig = model_cfg.get("ddconfig", {})
        preprocessing_cfg = audio_vae_cfg.get("preprocessing", {})
        stft_cfg = preprocessing_cfg.get("stft", {})
        mel_cfg = preprocessing_cfg.get("mel", {})

        sample_rate = int(model_cfg.get("sampling_rate", 16000))
        hop_length = int(stft_cfg.get("hop_length", 160))
        channels = int(ddconfig.get("z_channels", 8))
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or 64
        mel_bins = int(mel_bins)

        audio_patchify_proj = getattr(transformer, "audio_patchify_proj", None)
        audio_in_features = getattr(audio_patchify_proj, "in_features", None)
        if isinstance(audio_in_features, int) and channels > 0:
            inferred_mel = audio_in_features // channels
            if inferred_mel > 0 and inferred_mel != mel_bins:
                logger.warning(
                    "Sampling: overriding audio mel_bins from %s to %s to match audio_patchify_proj.in_features=%s",
                    mel_bins,
                    inferred_mel,
                    audio_in_features,
                )
                mel_bins = inferred_mel
            elif audio_in_features % channels != 0:
                logger.warning(
                    "Sampling: audio_patchify_proj.in_features=%s is not divisible by audio channels=%s; audio preview may fail.",
                    audio_in_features,
                    channels,
                )

        self._audio_preview_config = {
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "channels": channels,
            "mel_bins": mel_bins,
            "audio_latent_downsample_factor": int(LATENT_DOWNSAMPLE_FACTOR),
        }
        return self._audio_preview_config

    def _normalize_reference_tensor_collection(
        self,
        ref_value,
        *,
        expected_ndim: int,
    ) -> List[torch.Tensor]:
        refs: List[torch.Tensor] = []
        if ref_value is None:
            return refs
        if isinstance(ref_value, (list, tuple)):
            for item in ref_value:
                refs.extend(self._normalize_reference_tensor_collection(item, expected_ndim=expected_ndim))
            return refs
        if not isinstance(ref_value, torch.Tensor):
            return refs
        if ref_value.dim() == expected_ndim:
            refs.append(ref_value)
            return refs
        if ref_value.dim() == expected_ndim + 1:
            refs.extend([ref_value[:, i, ...] for i in range(int(ref_value.shape[1]))])
            return refs
        raise ValueError(
            f"Expected reference tensor with ndim {expected_ndim} or {expected_ndim + 1}, got {ref_value.dim()}"
        )

    def _load_audio_components(
        self,
        args: argparse.Namespace,
        audio_dtype: torch.dtype,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        logger.info("Loading LTX-2 audio decoder/vocoder from %s (device=%s)", checkpoint_path, device)
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AudioDecoderConfigurator,
            VocoderConfigurator,
            AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            VOCODER_COMFY_KEYS_FILTER,
        )

        audio_decoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)
        vocoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=audio_dtype)

        audio_decoder.eval()
        vocoder.eval()
        return audio_decoder, vocoder

    @staticmethod
    def _save_audio_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
        audio = audio.detach().cpu().float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if audio.shape[0] > 2:
            audio = audio[:2, :]
        audio_int16 = (audio.clamp(-1, 1) * 32767.0).to(torch.int16)
        interleaved = audio_int16.t().contiguous().numpy().tobytes()
        with wave.open(path, "wb") as wav:
            wav.setnchannels(audio_int16.shape[0])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(interleaved)

    def _decode_audio_preview_subprocess(
        self,
        *,
        audio_latents: torch.Tensor,
        output_path: str,
        checkpoint_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix="_ltx2_audio_latents.pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save({"latents": audio_latents.detach().cpu()}, tmp_path)
            cmd = [
                sys.executable,
                "-m",
                "musubi_tuner.ltx2_audio_preview",
                "--checkpoint",
                checkpoint_path,
                "--input",
                tmp_path,
                "--output",
                output_path,
                "--device",
                "auto",
                "--dtype",
                "fp32",
            ]
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            result = subprocess.run(cmd, capture_output=True, env=env)
            if result.returncode != 0:
                logger.warning(
                    "Audio preview subprocess failed (code=%s): %s",
                    result.returncode,
                    (_decode_subprocess_output(result.stderr) or _decode_subprocess_output(result.stdout)).strip(),
                )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _cleanup_cuda(device: torch.device) -> None:
        clean_memory_on_device(device)
        if device.type == "cuda":
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()

    @staticmethod
    def _mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return
        try:
            import av
            import numpy as np
        except Exception as exc:
            logger.warning("Sampling: unable to mux audio/video (PyAV missing?): %s", exc)
            return

        with wave.open(audio_path, "rb") as wav_in:
            sample_rate = wav_in.getframerate()
            channels = wav_in.getnchannels()
            frames = wav_in.readframes(wav_in.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        else:
            audio = audio.reshape(-1, 1)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        container_in = av.open(video_path)
        video_stream_in = next((s for s in container_in.streams if s.type == "video"), None)
        if video_stream_in is None:
            container_in.close()
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        container_out = av.open(output_path, mode="w")
        video_stream_out = container_out.add_stream(
            "libx264",
            rate=video_stream_in.average_rate or video_stream_in.base_rate or 24,
        )
        video_stream_out.width = video_stream_in.width
        video_stream_out.height = video_stream_in.height
        video_stream_out.pix_fmt = "yuv420p"

        audio_stream = container_out.add_stream("aac", rate=sample_rate)
        audio_stream.codec_context.sample_rate = sample_rate
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, sample_rate)

        for frame in container_in.decode(video_stream_in):
            for packet in video_stream_out.encode(frame):
                container_out.mux(packet)
        for packet in video_stream_out.encode():
            container_out.mux(packet)

        frame_in = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="stereo")
        frame_in.sample_rate = sample_rate
        target_format = audio_stream.codec_context.format or "fltp"
        target_layout = audio_stream.codec_context.layout or "stereo"
        target_rate = audio_stream.codec_context.sample_rate or sample_rate
        audio_resampler = av.audio.resampler.AudioResampler(
            format=target_format,
            layout=target_layout,
            rate=target_rate,
        )
        audio_next_pts = 0
        for rframe in audio_resampler.resample(frame_in):
            if rframe.pts is None:
                rframe.pts = audio_next_pts
            audio_next_pts += rframe.samples
            rframe.sample_rate = sample_rate
            for packet in audio_stream.encode(rframe):
                container_out.mux(packet)
        for packet in audio_stream.encode():
            container_out.mux(packet)

        container_out.close()
        container_in.close()

    @staticmethod
    def _override_attention_function(transformer, attention_function):
        from musubi_tuner.ltx_2.model.transformer.attention import Attention, AttentionFunction

        if isinstance(attention_function, AttentionFunction):
            attention_function = attention_function.to_callable()
        overrides = []
        for module in transformer.modules():
            if isinstance(module, Attention):
                overrides.append((module, module.attention_function))
                module.attention_function = attention_function
        return overrides

    @staticmethod
    def _restore_attention_function(overrides) -> None:
        for module, attention_function in overrides:
            module.attention_function = attention_function

    def _apply_sample_defaults(self, args: argparse.Namespace, prompts: List[Dict]) -> List[Dict]:
        preset = get_ltx2_sampling_preset(
            getattr(args, "sample_sampling_preset", "legacy"),
            ltx_version=str(getattr(args, "ltx_version", "2.3")),
        )
        default_height = int(getattr(args, "height", 512))
        default_width = int(getattr(args, "width", 768))
        default_frame_count = int(getattr(args, "sample_num_frames", 45))
        default_sample_steps = 20
        default_guidance_scale = float(getattr(args, "guidance_scale", self.default_guidance_scale))
        default_negative_prompt = ""
        if preset is not None:
            default_height = preset.height
            default_width = preset.width
            default_frame_count = preset.frame_count
            default_sample_steps = preset.sample_steps
            default_guidance_scale = preset.video_cfg_scale
            if getattr(args, "video_cfg_scale", None) is None:
                args.video_cfg_scale = preset.video_cfg_scale
            if getattr(args, "audio_cfg_scale", None) is None:
                args.audio_cfg_scale = preset.audio_cfg_scale
            if getattr(args, "stg_scale", None) is None:
                args.stg_scale = preset.stg_scale
            if getattr(args, "stg_blocks", None) is None:
                args.stg_blocks = preset.stg_blocks
            if getattr(args, "stg_mode", None) is None:
                args.stg_mode = preset.stg_mode
            if getattr(args, "video_rescale_scale", None) is None:
                args.video_rescale_scale = preset.video_rescale_scale
            if getattr(args, "audio_rescale_scale", None) is None:
                args.audio_rescale_scale = preset.audio_rescale_scale
            if getattr(args, "video_modality_scale", None) is None:
                args.video_modality_scale = preset.video_modality_scale
            if getattr(args, "audio_modality_scale", None) is None:
                args.audio_modality_scale = preset.audio_modality_scale
            if getattr(args, "av_bimodal_cfg", False) is False:
                args.av_bimodal_cfg = preset.video_modality_scale != 1.0 or preset.audio_modality_scale != 1.0
            if getattr(args, "av_bimodal_scale", 3.0) == 3.0:
                args.av_bimodal_scale = preset.video_modality_scale
            use_default_negative = getattr(args, "sample_use_default_negative_prompt", None)
            if use_default_negative is None or bool(use_default_negative):
                default_negative_prompt = preset.negative_prompt
        if getattr(args, "stg_scale", None) is None:
            args.stg_scale = 0.0
        if getattr(args, "stg_mode", None) is None:
            args.stg_mode = "video"
        default_discrete_flow_shift = getattr(args, "discrete_flow_shift", None)

        sample_parameters = []
        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt", "")
            param = prompt_data.copy()
            param.setdefault("prompt", prompt_text)
            param.setdefault("negative_prompt", prompt_data.get("negative_prompt", default_negative_prompt))
            if "frame_count" not in param and "num_frames" in param:
                param["frame_count"] = param["num_frames"]
            param.setdefault("height", prompt_data.get("height", default_height))
            param.setdefault("width", prompt_data.get("width", default_width))
            param.setdefault("frame_count", prompt_data.get("frame_count", default_frame_count))
            param.setdefault("sample_steps", prompt_data.get("sample_steps", default_sample_steps))
            param.setdefault("guidance_scale", prompt_data.get("guidance_scale", default_guidance_scale))
            if default_discrete_flow_shift is not None:
                param.setdefault("discrete_flow_shift", prompt_data.get("discrete_flow_shift", default_discrete_flow_shift))
            param.setdefault("seed", prompt_data.get("seed", 0))
            sample_parameters.append(param)

        return sample_parameters

    def _load_precached_sample_prompts(self, args: argparse.Namespace) -> List[Dict]:
        cache_path = getattr(args, "sample_prompts_cache", None) or self._resolve_default_sample_prompts_cache(args)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Precached sample prompt embeddings not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sample prompt cache format: {cache_path}")
        cached_params = payload.get("prompt_cache") or payload.get("sample_parameters")
        if not isinstance(cached_params, list) or not cached_params:
            raise ValueError(f"No sample prompts found in cache: {cache_path}")

        if args.sample_prompts is None:
            raise ValueError("--sample_prompts is required when --use_precached_sample_prompts is set")
        prompts = load_prompts(args.sample_prompts)
        if not prompts:
            raise ValueError(f"No prompts found in {args.sample_prompts}")

        sample_params = self._apply_sample_defaults(args, prompts)
        if len(sample_params) != len(cached_params):
            raise ValueError(
                "Sample prompt count does not match precached embeddings "
                f"(prompts={len(sample_params)} cache={len(cached_params)})."
            )

        def _normalize_text(value: Optional[str]) -> str:
            if value is None:
                return ""
            return " ".join(str(value).split())

        for idx, param in enumerate(sample_params):
            cache_entry = cached_params[idx]
            if not isinstance(cache_entry, dict):
                raise ValueError(f"Invalid cache entry at {idx} ({cache_path})")

            cfg_scale = param.get("cfg_scale", None)
            guidance_scale = param.get("guidance_scale", self.default_guidance_scale)
            effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
            video_cfg_scale = getattr(args, "video_cfg_scale", None)
            audio_cfg_scale = getattr(args, "audio_cfg_scale", None)
            try:
                requires_negative_embed = (
                    float(effective_cfg_scale) != 1.0
                    or (video_cfg_scale is not None and float(video_cfg_scale) != 1.0)
                    or (audio_cfg_scale is not None and float(audio_cfg_scale) != 1.0)
                )
            except (TypeError, ValueError):
                requires_negative_embed = False

            expected_prompt = _normalize_text(param.get("prompt", ""))
            cached_prompt = _normalize_text(cache_entry.get("prompt", ""))
            if expected_prompt != cached_prompt:
                raise ValueError(
                    "Prompt text mismatch with precached embeddings at index "
                    f"{idx} ({cache_path}). Rebuild sample prompt cache or disable "
                    "--use_precached_sample_prompts.\n"
                    f"Current: {param.get('prompt', '')}\n"
                    f"Cached : {cache_entry.get('prompt', '')}"
                )

            expected_negative = _normalize_text(param.get("negative_prompt", ""))
            cached_negative = _normalize_text(cache_entry.get("negative_prompt", ""))
            if expected_negative != cached_negative:
                raise ValueError(
                    "Negative prompt mismatch with precached embeddings at index "
                    f"{idx} ({cache_path}). Rebuild sample prompt cache or disable "
                    "--use_precached_sample_prompts.\n"
                    f"Current: {param.get('negative_prompt', '')}\n"
                    f"Cached : {cache_entry.get('negative_prompt', '')}"
                )

            if cache_entry.get("prompt_embeds") is None or cache_entry.get("prompt_attention_mask") is None:
                raise ValueError(f"Missing prompt embeddings in cache entry {idx} ({cache_path})")
            param["prompt_embeds"] = cache_entry["prompt_embeds"]
            param["prompt_attention_mask"] = cache_entry["prompt_attention_mask"]
            if requires_negative_embed or param.get("negative_prompt"):
                if cache_entry.get("negative_prompt_embeds") is None or cache_entry.get(
                    "negative_prompt_attention_mask"
                ) is None:
                    raise ValueError(
                        "Missing negative prompt embeddings in cache entry "
                        f"{idx} ({cache_path}); this prompt needs CFG (guidance/cfg != 1), "
                        "so negative embeddings must be precached."
                    )
                param["negative_prompt_embeds"] = cache_entry["negative_prompt_embeds"]
                param["negative_prompt_attention_mask"] = cache_entry["negative_prompt_attention_mask"]

        return sample_params

    def _load_precached_sample_latents(self, args: argparse.Namespace, sample_params: List[Dict]) -> None:
        """Load precached I2V / V2V / reference-audio latents into sample_params (in-place)."""
        cache_path = getattr(args, "sample_latents_cache", None) or self._resolve_default_sample_latents_cache(args)
        if not os.path.exists(cache_path):
            logger.warning("Precached latents not found: %s — skipping (samples will run without conditioning)", cache_path)
            return

        logger.info(f"Loading precached conditioning latents from {cache_path}")
        try:
            latent_payload = torch.load(cache_path, map_location="cpu")
            latent_cache = latent_payload.get("latent_cache", [])

            # Match latents with prompts by index
            i2v_count = 0
            v2v_count = 0
            ref_audio_count = 0
            for entry in latent_cache:
                prompt_idx = entry.get("prompt_index")
                if prompt_idx is not None and 0 <= prompt_idx < len(sample_params):
                    if "conditioning_latent" in entry:
                        sample_params[prompt_idx]["conditioning_latent"] = entry["conditioning_latent"]
                        i2v_count += 1
                    if "v2v_ref_latent" in entry:
                        sample_params[prompt_idx]["v2v_ref_latent"] = entry["v2v_ref_latent"]
                        v2v_count += 1
                    ref_audio_latent_entry = entry.get("ref_audio_latent")
                    if ref_audio_latent_entry is None and "reference_audio_latent" in entry:
                        ref_audio_latent_entry = entry["reference_audio_latent"]
                    if ref_audio_latent_entry is not None:
                        sample_params[prompt_idx]["ref_audio_latent"] = ref_audio_latent_entry
                        ref_audio_count += 1

                    ref_audio_path_entry = entry.get("ref_audio_path")
                    if ref_audio_path_entry is None and "reference_audio_path" in entry:
                        ref_audio_path_entry = entry["reference_audio_path"]
                    if ref_audio_path_entry is not None and "ref_audio_path" not in sample_params[prompt_idx]:
                        sample_params[prompt_idx]["ref_audio_path"] = ref_audio_path_entry

            logger.info(
                "Loaded precached latents: %d I2V, %d V2V references, %d reference-audio",
                i2v_count,
                v2v_count,
                ref_audio_count,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load latents cache: {e}")

    def _resolve_first_dataset_cache_directory(self, args: argparse.Namespace) -> str:
        from musubi_tuner.dataset import config_utils
        from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        if getattr(args, "dataset_manifest", None):
            dataset_manifest = config_utils.load_dataset_manifest(args.dataset_manifest)
            manifest_architecture = dataset_manifest.get("architecture")
            if manifest_architecture is not None and manifest_architecture != ARCHITECTURE_LTX2:
                raise ValueError(
                    f"dataset manifest architecture mismatch: expected '{ARCHITECTURE_LTX2}', got '{manifest_architecture}'"
                )
            datasets = dataset_manifest.get("datasets", [])
            if not datasets:
                raise ValueError("No datasets available in dataset manifest to resolve sample cache directory")
            cache_dir = datasets[0].get("params", {}).get("cache_directory")
            if not cache_dir:
                raise ValueError("First manifest dataset has no cache_directory")
            return str(cache_dir)

        if getattr(args, "dataset_config", None):
            user_config = config_utils.load_user_config(args.dataset_config)
            blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, args, architecture=ARCHITECTURE_LTX2)
            dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
            datasets = dataset_group.datasets
            if not datasets:
                raise ValueError("No datasets available to resolve sample cache directory")
            cache_dir = getattr(datasets[0], "cache_directory", None)
            if not cache_dir:
                raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
            return cache_dir

        raise ValueError("--dataset_config or --dataset_manifest is required to resolve sample cache directory")

    def _resolve_default_sample_prompts_cache(self, args: argparse.Namespace) -> str:
        cache_dir = self._resolve_first_dataset_cache_directory(args)
        return os.path.join(cache_dir, DEFAULT_SAMPLE_PROMPTS_CACHE)

    def _resolve_default_sample_latents_cache(self, args: argparse.Namespace) -> str:
        """Resolve default path for sample latents cache (same directory as prompts cache)."""
        cache_dir = self._resolve_first_dataset_cache_directory(args)
        return os.path.join(cache_dir, DEFAULT_SAMPLE_LATENTS_CACHE)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ) -> Optional[List[Dict]]:
        """Process sample prompts for inference preview during training"""
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        if use_precached:
            logger.info("LTX-2 sampling: using precached Gemma embeddings for sample prompts")
            sample_params = self._load_precached_sample_prompts(args)
        else:
            logger.info("LTX-2 sampling: deferring Gemma encoding until sampling")
            prompts = load_prompts(sample_prompts)
            if not prompts:
                return None
            sample_params = self._apply_sample_defaults(args, prompts)

        # Load precached I2V latents if requested (independent of text embedding caching)
        use_precached_latents = bool(getattr(args, "use_precached_sample_latents", False))
        if use_precached_latents:
            logger.info("LTX-2 sampling: using precached I2V conditioning latents")
            self._load_precached_sample_latents(args, sample_params)

        return sample_params

    def _build_text_encoder(self, args: argparse.Namespace, accelerator: Accelerator) -> torch.dtype:
        logger.info("Loading Gemma text encoder for LTX-2 sampling")
        gemma_safetensors = getattr(args, "gemma_safetensors", None)
        if getattr(args, "gemma_root", None) is None and not gemma_safetensors:
            raise ValueError("--gemma_root or --gemma_safetensors is required for LTX-2 sample prompts")
        if getattr(args, "ltx2_checkpoint", None) is None:
            raise ValueError("--ltx2_checkpoint is required for LTX-2 sample prompts")
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.av_encoder import (
            AVGemmaTextEncoderModelConfigurator,
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        )
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import (
            apply_text_encoder_checkpoint_overrides,
            module_ops_from_gemma_root,
        )
        from musubi_tuner.ltx_2.text_encoders.gemma.encoders.video_only_encoder import (
            VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
            VideoGemmaTextEncoderModelConfigurator,
        )

        configurator = AVGemmaTextEncoderModelConfigurator if self._audio_video else VideoGemmaTextEncoderModelConfigurator
        key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if self._audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

        mixed_precision = getattr(accelerator, "mixed_precision", "no")
        if mixed_precision == "bf16":
            text_encoder_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            text_encoder_dtype = torch.float16
        else:
            text_encoder_dtype = torch.float32

        if getattr(args, "gemma_load_in_8bit", False) or getattr(args, "gemma_load_in_4bit", False):
            if accelerator.device.type != "cuda":
                raise ValueError("Gemma 8-bit/4-bit loading requires CUDA")

        build_device = accelerator.device
        is_quantized_load = getattr(args, "gemma_load_in_8bit", False) or getattr(args, "gemma_load_in_4bit", False)

        self._text_encoder = SingleGPUModelBuilder(
            model_path=str(args.ltx2_checkpoint),
            model_class_configurator=configurator,
            model_sd_ops=key_ops,
            module_ops=module_ops_from_gemma_root(
                args.gemma_root,
                gemma_safetensors=gemma_safetensors,
                torch_dtype=text_encoder_dtype,
                load_in_8bit=bool(getattr(args, "gemma_load_in_8bit", False)),
                load_in_4bit=bool(getattr(args, "gemma_load_in_4bit", False)),
                bnb_4bit_quant_type=str(getattr(args, "gemma_bnb_4bit_quant_type", "nf4")),
                bnb_4bit_use_double_quant=not bool(getattr(args, "gemma_bnb_4bit_disable_double_quant", False)),
                bnb_4bit_compute_dtype=text_encoder_dtype,
                fp8_weight_offload=getattr(args, "gemma_fp8_weight_offload", None),
                device=build_device,
            ),
        ).build(device=build_device, dtype=text_encoder_dtype)
        text_model = getattr(self._text_encoder, "model", None)
        is_quantized = False
        if text_model is not None:
            is_quantized = bool(getattr(text_model, "is_loaded_in_8bit", False)) or bool(
                getattr(text_model, "is_loaded_in_4bit", False)
            )
        is_fp8 = bool(getattr(self._text_encoder, "_has_fp8_model", False))
        if not is_quantized and not is_fp8 and accelerator.device.type != "cpu":
            self._text_encoder.to(accelerator.device)
        text_model = getattr(self._text_encoder, "model", None)
        if text_model is not None:
            try:
                first_param = next(text_model.parameters())
                logger.info(
                    "Gemma text encoder device: %s dtype: %s",
                    first_param.device,
                    first_param.dtype,
                )
            except StopIteration:
                pass
        apply_text_encoder_checkpoint_overrides(self._text_encoder, str(args.ltx2_checkpoint))
        self._text_encoder.eval()

        # Connector LoRA: replace text encoder's connectors with the wrapper's
        # (which have LoRA hooks or merged weights), so samples reflect connector LoRA.
        # This covers both training (--train_connectors) and inference (merged connector LoRA).
        transformer = getattr(self, "unet", None)
        if transformer is not None and getattr(transformer, "has_connectors", lambda: False)():
            if hasattr(self._text_encoder, "embeddings_connector"):
                self._text_encoder.embeddings_connector = transformer.embeddings_connector
                logger.info("Sampling: using LoRA'd video connector from wrapper")
            if hasattr(self._text_encoder, "audio_embeddings_connector") and hasattr(transformer, "audio_embeddings_connector"):
                self._text_encoder.audio_embeddings_connector = transformer.audio_embeddings_connector
                logger.info("Sampling: using LoRA'd audio connector from wrapper")

        return text_encoder_dtype

    def _encode_prompt_text(
        self,
        accelerator: Accelerator,
        prompt_text: str,
        text_encoder_dtype: torch.dtype,
        *,
        allow_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep gradients for full-FT text-encoder training when explicitly requested.
        if allow_grad:
            with accelerator.autocast():
                out = self._text_encoder(prompt_text, padding_side="left")
                if self._ltx_mode == "audio":
                    embed = out.audio_encoding if hasattr(out, "audio_encoding") else out.video_encoding
                elif self._audio_video:
                    embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
                else:
                    embed = out.video_encoding
                mask = out.attention_mask
            return embed.squeeze(0).to(dtype=text_encoder_dtype), mask.squeeze(0).to(dtype=out.attention_mask.dtype)

        with accelerator.autocast(), torch.no_grad():
            out = self._text_encoder(prompt_text, padding_side="left")
            if self._ltx_mode == "audio":
                embed = out.audio_encoding if hasattr(out, "audio_encoding") else out.video_encoding
            elif self._audio_video:
                embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
            else:
                embed = out.video_encoding
            mask = out.attention_mask
        return embed.squeeze(0).detach().cpu(), mask.squeeze(0).detach().cpu()

    def _cleanup_text_encoder(self, accelerator: Accelerator) -> None:
        if self._text_encoder is None:
            return
        if hasattr(self._text_encoder, "model"):
            self._text_encoder.model = None
        if hasattr(self._text_encoder, "tokenizer"):
            self._text_encoder.tokenizer = None
        if hasattr(self._text_encoder, "feature_extractor_linear"):
            self._text_encoder.feature_extractor_linear = None
        self._text_encoder = None
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

    def sample_images(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        epoch,
        steps,
        vae,
        transformer,
        sample_parameters,
        dit_dtype,
    ):
        """LTX-2 sampling with optional DiT offloading between prompts."""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
        if sample_parameters is None:
            if getattr(args, "use_precached_sample_prompts", False) or getattr(args, "precache_sample_prompts", False):
                logger.error("No precached sample prompt embeddings found. Check --sample_prompts_cache.")
            else:
                logger.error(f"No prompt file / ???????????????: {args.sample_prompts}")
            return

        distributed_state = PartialState()  # for multi gpu distributed inference

        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()
        original_device = next(transformer.parameters()).device
        offload = bool(getattr(args, "sample_with_offloading", False))
        transformer_offloaded = offload and accelerator.device.type == "cuda"
        if transformer_offloaded:
            transformer.to("cpu")
            logger.info("Sampling offload: moved transformer to CPU before prompt loop")
            clean_memory_on_device(accelerator.device)
        if getattr(transformer, "blocks_to_swap", 0) and original_device.type == "cpu" and not transformer_offloaded:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            else:
                transformer.to(accelerator.device)
            clean_memory_on_device(accelerator.device)
            original_device = accelerator.device

        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        def ensure_transformer_on_device() -> None:
            if transformer_offloaded:
                logger.info("Sampling offload: moving transformer to GPU for denoise")
                if hasattr(transformer, "move_to_device_except_swap_blocks"):
                    transformer.move_to_device_except_swap_blocks(accelerator.device)
                else:
                    transformer.to(accelerator.device)
                clean_memory_on_device(accelerator.device)

        def offload_transformer_if_needed() -> None:
            if transformer_offloaded:
                logger.info("Sampling offload: moving transformer back to CPU")
                transformer.to("cpu")
                clean_memory_on_device(accelerator.device)

        def cleanup_embeddings(sample_parameter: Dict) -> None:
            sample_parameter.pop("prompt_embeds", None)
            sample_parameter.pop("prompt_attention_mask", None)
            sample_parameter.pop("negative_prompt_embeds", None)
            sample_parameter.pop("negative_prompt_attention_mask", None)

        def prepare_all_embeddings_batch(sample_params_list: List[Dict]) -> None:
            """Load text encoder once and encode ALL prompts before unloading."""
            def _requires_negative_embeddings(sample_parameter: Dict) -> bool:
                cfg_scale = sample_parameter.get("cfg_scale", None)
                guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
                effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
                video_cfg_scale = getattr(args, "video_cfg_scale", None)
                audio_cfg_scale = getattr(args, "audio_cfg_scale", None)
                try:
                    return (
                        float(effective_cfg_scale) != 1.0
                        or (video_cfg_scale is not None and float(video_cfg_scale) != 1.0)
                        or (audio_cfg_scale is not None and float(audio_cfg_scale) != 1.0)
                    )
                except (TypeError, ValueError):
                    return False

            missing_indices = []
            for idx, sample_parameter in enumerate(sample_params_list):
                needs_prompt = sample_parameter.get("prompt_embeds") is None
                needs_negative = _requires_negative_embeddings(sample_parameter) and sample_parameter.get(
                    "negative_prompt_embeds"
                ) is None
                if needs_prompt or needs_negative:
                    missing_indices.append(idx)

            if not missing_indices:
                return

            strict_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
                getattr(args, "precache_sample_prompts", False)
            )
            if strict_precached:
                preview = ",".join(str(i) for i in missing_indices[:10])
                if len(missing_indices) > 10:
                    preview += ",..."
                raise ValueError(
                    "Precached sample prompt embeddings are incomplete; refusing to load Gemma during training. "
                    f"Missing prompt/negative embeddings for sample indices [{preview}]. "
                    "Rebuild sample prompt cache with ltx2_cache_text_encoder_outputs.py."
                )

            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            logger.info("Sampling batch: loaded text encoder for %d prompts", len(sample_params_list))

            for sample_parameter in sample_params_list:
                if sample_parameter.get("prompt_embeds") is None:
                    prompt_text = sample_parameter.get("prompt", "")
                    prompt_embeds, prompt_mask = self._encode_prompt_text(accelerator, prompt_text, text_encoder_dtype)
                    sample_parameter["prompt_embeds"] = prompt_embeds
                    sample_parameter["prompt_attention_mask"] = prompt_mask

                if _requires_negative_embeddings(sample_parameter) and sample_parameter.get("negative_prompt_embeds") is None:
                    negative_prompt = sample_parameter.get("negative_prompt")
                    if negative_prompt is None:
                        negative_prompt = ""
                        sample_parameter["negative_prompt"] = negative_prompt
                    neg_embeds, neg_mask = self._encode_prompt_text(accelerator, negative_prompt, text_encoder_dtype)
                    sample_parameter["negative_prompt_embeds"] = neg_embeds
                    sample_parameter["negative_prompt_attention_mask"] = neg_mask

            self._cleanup_text_encoder(accelerator)
            logger.info("Sampling batch: unloaded text encoder after encoding all prompts")
            self._cleanup_cuda(accelerator.device)

        # Check if using precached prompts (don't cleanup precached embeddings - they're reused)
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )

        # Pre-load audio components only in NON-offloading mode without subprocess (high VRAM)
        # With subprocess (default): audio decoded in separate process, no in-process loading needed
        # With offloading: audio will be decoded via subprocess during decode phase
        audio_decoder = None
        vocoder = None
        use_audio_subprocess = bool(getattr(args, "sample_audio_subprocess", True))
        disable_audio_preview = bool(getattr(args, "sample_disable_audio", False))
        audio_only_preview = bool(getattr(args, "sample_audio_only", False))
        if self._ltx_mode == "audio":
            audio_only_preview = True
        enable_audio_preview = (self._audio_video or audio_only_preview) and not disable_audio_preview
        if not transformer_offloaded and not use_audio_subprocess and enable_audio_preview and getattr(args, "ltx_mode", "video") in {"av", "audio"}:
            # High VRAM mode without subprocess: pre-load audio to GPU
            audio_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
            try:
                audio_decoder, vocoder = self._load_audio_components(
                    args,
                    audio_dtype=audio_dtype,
                    checkpoint_path=args.ltx2_checkpoint,
                    device=accelerator.device,
                )
                logger.info("Sampling: pre-loaded audio decoder/vocoder to GPU (high VRAM mode)")
            except Exception as exc:
                logger.warning("Sampling audio decoder load failed; continuing without audio preview: %s", exc)
                audio_decoder, vocoder = None, None

        if distributed_state.num_processes <= 1:
            # Batch encode all prompts upfront when offloading is enabled
            if transformer_offloaded:
                offload_transformer_if_needed()
                prepare_all_embeddings_batch(sample_parameters)

            # Load VAE once before the prompt loop to avoid repeated disk reads from the
            # (potentially huge) safetensors checkpoint.  Keep it on CPU between prompts.
            vae_for_sampling = None
            if transformer_offloaded:
                vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                logger.info("Sampling offload: loading VAE for sampling (once)")
                vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)

            with torch.no_grad(), accelerator.autocast():
                for sample_parameter in sample_parameters:
                    try:
                        if transformer_offloaded:
                            ensure_transformer_on_device()
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae_for_sampling, save_dir, sample_parameter, epoch, steps,
                                audio_decoder=audio_decoder, vocoder=vocoder,
                            )
                            offload_transformer_if_needed()
                            vae_for_sampling.to_device("cpu")
                            self._cleanup_cuda(accelerator.device)
                        else:
                            self.sample_image_inference(
                                accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps,
                                audio_decoder=audio_decoder, vocoder=vocoder,
                            )
                    except Exception as exc:
                        logger.error("Sampling failed for prompt, skipping: %s", exc, exc_info=True)
                    clean_memory_on_device(accelerator.device)
                    self._cleanup_cuda(accelerator.device)

            if vae_for_sampling is not None:
                del vae_for_sampling
                self._cleanup_cuda(accelerator.device)

            # Cleanup embeddings after all samples are done (but NOT if precached - they're reused)
            if transformer_offloaded and not use_precached:
                for sample_parameter in sample_parameters:
                    cleanup_embeddings(sample_parameter)
        else:
            per_process_params = []
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    my_sample_params = sample_parameter_lists[0]

                    # Batch encode all prompts for this process upfront
                    if transformer_offloaded:
                        offload_transformer_if_needed()
                        prepare_all_embeddings_batch(my_sample_params)

                    # Load VAE once before the prompt loop
                    vae_for_sampling = None
                    if transformer_offloaded:
                        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                        logger.info("Sampling offload: loading VAE for sampling (once)")
                        vae_for_sampling = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)

                    for sample_parameter in my_sample_params:
                        try:
                            if transformer_offloaded:
                                ensure_transformer_on_device()
                                self.sample_image_inference(
                                    accelerator,
                                    args,
                                    transformer,
                                    dit_dtype,
                                    vae_for_sampling,
                                    save_dir,
                                    sample_parameter,
                                    epoch,
                                    steps,
                                    audio_decoder=audio_decoder,
                                    vocoder=vocoder,
                                )
                                offload_transformer_if_needed()
                                vae_for_sampling.to_device("cpu")
                                self._cleanup_cuda(accelerator.device)
                            else:
                                self.sample_image_inference(
                                    accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps,
                                    audio_decoder=audio_decoder, vocoder=vocoder,
                                )
                        except Exception as exc:
                            logger.error("Sampling failed for prompt, skipping: %s", exc, exc_info=True)
                        self._cleanup_cuda(accelerator.device)

                    if vae_for_sampling is not None:
                        del vae_for_sampling
                        self._cleanup_cuda(accelerator.device)

                    # Cleanup embeddings after all samples for this process (but NOT if precached)
                    if transformer_offloaded and not use_precached:
                        for sample_parameter in my_sample_params:
                            cleanup_embeddings(sample_parameter)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        if transformer_offloaded and next(transformer.parameters()).device != accelerator.device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            else:
                transformer.to(accelerator.device)
            logger.info("Sampling offload: restored transformer to training device")
            clean_memory_on_device(accelerator.device)

        transformer.switch_block_swap_for_training()
        # Ensure block-swap layout is re-applied after sampling to avoid VRAM creep.
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(accelerator.device)
        self._cleanup_cuda(accelerator.device)

    @staticmethod
    def _load_reference_for_output(
        ref_path: str,
        target_height: int,
        target_width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Load reference image/video as [1, C, T, H, W] in [0,1] for side-by-side output."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        ext = os.path.splitext(ref_path)[1].lower()
        is_video = ext in [e.lower() for e in VIDEO_EXTENSIONS]

        def _cover_center_crop_out(pil_img, tw, th):
            cw, ch = pil_img.size
            if ch == th and cw == tw:
                return pil_img
            ar = cw / ch
            tar = tw / th
            if ar > tar:
                rh = th
                rw = max(tw, int(round(th * ar)))
            else:
                rw = tw
                rh = max(th, int(round(tw / ar)))
            pil_img = pil_img.resize((rw, rh), Image.LANCZOS)
            left = max((rw - tw) // 2, 0)
            top = max((rh - th) // 2, 0)
            return pil_img.crop((left, top, left + tw, top + th))

        frames = []
        if is_video:
            try:
                import av
                container = av.open(ref_path)
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= num_frames:
                        break
                    pil_frame = _cover_center_crop_out(frame.to_image().convert("RGB"), target_width, target_height)
                    frames.append(TF.to_tensor(pil_frame))
                container.close()
            except Exception as e:
                logger.warning(f"Failed to load reference video for output: {e}")
        if not frames:
            image = _cover_center_crop_out(Image.open(ref_path).convert("RGB"), target_width, target_height)
            frames = [TF.to_tensor(image)]

        while len(frames) < num_frames:
            frames.append(frames[-1])
        frames = frames[:num_frames]

        video = torch.stack(frames, dim=1).unsqueeze(0)
        return video.clamp(0, 1).to(torch.float32)

    def _load_and_encode_v2v_reference(
        self,
        ref_path: str,
        target_height: int,
        target_width: int,
        vae_checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
        max_frames: int = 1,
    ) -> torch.Tensor:
        """Load image or video from disk and encode through VAE for V2V reference conditioning.

        Returns:
            Encoded latent tensor [1, C, F, H_latent, W_latent]
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"V2V reference not found: {ref_path}")

        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        ext = os.path.splitext(ref_path)[1].lower()
        is_video = ext in {e.lower() for e in VIDEO_EXTENSIONS}

        def _cover_center_crop(pil_img, tw, th):
            cw, ch = pil_img.size
            if ch == th and cw == tw:
                return pil_img
            ar = cw / ch
            tar = tw / th
            if ar > tar:
                rh = th
                rw = max(tw, int(round(th * ar)))
            else:
                rw = tw
                rh = max(th, int(round(tw / ar)))
            pil_img = pil_img.resize((rw, rh), Image.LANCZOS)
            left = max((rw - tw) // 2, 0)
            top = max((rh - th) // 2, 0)
            return pil_img.crop((left, top, left + tw, top + th))

        frames = []
        if is_video:
            import av
            container = av.open(ref_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_frames:
                    break
                pil_frame = _cover_center_crop(frame.to_image().convert("RGB"), target_width, target_height)
                frames.append(TF.to_tensor(pil_frame))
            container.close()
            if not frames:
                raise ValueError(f"No frames decoded from V2V reference video: {ref_path}")
        else:
            image = _cover_center_crop(Image.open(ref_path).convert("RGB"), target_width, target_height)
            frames.append(TF.to_tensor(image))

        # [F, 3, H, W] → [1, 3, F, H, W], normalize to [-1, 1]
        video_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # [1, F, 3, H, W]
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()  # [1, 3, F, H, W]
        video_tensor = (video_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

        # Pad frames to VAE alignment (LTX-2 VAE needs (F-1) % 8 == 0)
        num_frames = video_tensor.shape[2]
        remainder = (num_frames - 1) % 8
        if remainder != 0:
            pad = 8 - remainder
            last = video_tensor[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
            video_tensor = torch.cat([video_tensor, last], dim=2)

        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        logger.info("Loading VAE encoder for V2V reference")
        vae_encoder = SingleGPUModelBuilder(
            model_path=str(vae_checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        vae_encoder.eval()

        with torch.no_grad():
            latent = vae_encoder(video_tensor)  # [1, C, F_latent, H_latent, W_latent]

        logger.info(f"V2V reference encoded: {ref_path} → {latent.shape}")

        del vae_encoder
        clean_memory_on_device(device)

        return latent

    def _load_and_encode_conditioning_image(
        self,
        image_path: str,
        target_height: int,
        target_width: int,
        vae_checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load image from disk and encode through VAE for I2V conditioning.

        Args:
            image_path: Path to conditioning image (absolute or relative to working directory)
            target_height: Target video height (image will be resized)
            target_width: Target video width (image will be resized)
            vae_checkpoint_path: Path to VAE checkpoint
            device: Target device
            dtype: Target dtype

        Returns:
            Encoded image latent tensor [1, C, 1, H_latent, W_latent]
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"I2V conditioning image not found: {image_path}")

        logger.info(f"Loading I2V conditioning image: {image_path}")

        # Load and resize image with LTX "cover + center crop" behavior.
        # This preserves aspect ratio (unlike direct resize) and matches LTX-2 validation sampler.
        image = Image.open(image_path).convert("RGB")
        current_width, current_height = image.size
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                resize_height = target_height
                resize_width = max(target_width, int(round(target_height * aspect_ratio)))
            else:
                resize_width = target_width
                resize_height = max(target_height, int(round(target_width / aspect_ratio)))

            image = image.resize((resize_width, resize_height), Image.LANCZOS)
            left = max((resize_width - target_width) // 2, 0)
            top = max((resize_height - target_height) // 2, 0)
            image = image.crop((left, top, left + target_width, top + target_height))

        # Convert to tensor and normalize to [-1, 1]
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = (image_tensor * 2.0 - 1.0).to(device=device, dtype=dtype)

        # Add temporal dimension for VAE encoder: [B, C, T, H, W]
        image_tensor = image_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

        # Load VAE encoder (training only loads decoder, we need encoder for I2V)
        # Same approach as ltx2_cache_latents.py
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

        logger.info("Loading VAE encoder for I2V conditioning")
        vae_encoder = SingleGPUModelBuilder(
            model_path=str(vae_checkpoint_path),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        vae_encoder.eval()

        # Encode through VAE encoder
        with torch.no_grad():
            latent = vae_encoder(image_tensor)  # [1, C, 1, H_latent, W_latent]

        logger.info(f"Encoded I2V conditioning image to latent shape: {latent.shape}")

        # Clean up encoder to free VRAM
        del vae_encoder
        clean_memory_on_device(device)

        return latent

    def _load_and_encode_reference_audio_latent(
        self,
        audio_path: str,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load reference audio and encode it to LTX-2 audio latents [1, C, T, F]."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        try:
            import torchaudio
        except Exception as e:
            raise RuntimeError("torchaudio is required for reference-audio sampling") from e

        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioEncoderConfigurator,
        )
        from musubi_tuner.ltx_2.model.audio_vae.ops import AudioProcessor

        logger.info("Loading reference audio: %s", audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape from {audio_path}: {tuple(waveform.shape)}")

        channels = int(waveform.shape[0])
        if channels == 1:
            waveform = waveform.repeat(2, 1)
        elif channels == 2:
            pass
        elif channels > 2:
            mono = waveform.float().mean(dim=0, keepdim=True)
            waveform = mono.repeat(2, 1)

        encoder = SingleGPUModelBuilder(
            model_path=str(checkpoint_path),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        ).build(device=device, dtype=dtype)
        encoder.eval()

        processor = AudioProcessor(
            sample_rate=int(getattr(encoder, "sample_rate", 16000)),
            mel_bins=int(getattr(encoder, "mel_bins", 64)),
            mel_hop_length=int(getattr(encoder, "mel_hop_length", 160)),
            n_fft=int(getattr(encoder, "n_fft", 1024)),
        ).to(device=device, dtype=torch.float32)
        processor.eval()

        try:
            waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
            encoder_dtype = next(encoder.parameters()).dtype
            with torch.no_grad():
                mel = processor.waveform_to_mel(waveform, int(sample_rate)).to(device=device, dtype=encoder_dtype)
                latents = encoder(mel)
            latents = latents[0].detach().to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
            logger.info("Encoded reference audio latent shape: %s", tuple(latents.shape))
            return latents
        finally:
            del encoder
            del processor
            clean_memory_on_device(device)

    def sample_image_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        transformer,
        dit_dtype: torch.dtype,
        vae,
        save_dir: str,
        sample_parameter: Dict,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        """LTX-2-specific sampling with proper frame/size rounding."""

        # ===== PHASE 1: I2V Image Encoding (if needed) =====
        # Do this FIRST, before loading any other models, to respect --sample_with_offloading
        # This ensures only VAE encoder is in VRAM during encoding, then it's cleaned up completely
        conditioning_latent = None
        image_path = sample_parameter.get("image_path", None)

        # Check if we have a precached conditioning latent first
        if "conditioning_latent" in sample_parameter:
            conditioning_latent = sample_parameter["conditioning_latent"]
            if conditioning_latent is not None:
                device = accelerator.device
                conditioning_latent = conditioning_latent.to(device=device, dtype=dit_dtype)
                logger.info(f"I2V: Using precached conditioning latent (shape: {conditioning_latent.shape})")
                image_path = None  # Skip encoding since we have precached latent

        if image_path:
            logger.info("I2V: encoding conditioning image")
            try:
                vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
                if not vae_checkpoint:
                    raise ValueError("VAE checkpoint path required for I2V conditioning (--vae or --ltx2_checkpoint)")

                device = accelerator.device
                spatial_factor = 32
                temporal_factor = 8
                width = sample_parameter.get("width", 768)
                height = sample_parameter.get("height", 512)
                width = (width // spatial_factor) * spatial_factor
                height = (height // spatial_factor) * spatial_factor

                conditioning_latent = self._load_and_encode_conditioning_image(
                    image_path=image_path,
                    target_height=height,
                    target_width=width,
                    vae_checkpoint_path=vae_checkpoint,
                    device=device,
                    dtype=dit_dtype,
                )
                logger.info("I2V: conditioning image encoded")
            except Exception as e:
                logger.error(f"I2V: failed to load conditioning image '{image_path}': {e}")
                conditioning_latent = None

        v2v_ref_latent = None
        v2v_ref_path = sample_parameter.get("v2v_ref_path", None)

        if "v2v_ref_latent" in sample_parameter:
            v2v_ref_latent = sample_parameter["v2v_ref_latent"]
            if v2v_ref_latent is not None:
                device = accelerator.device
                v2v_ref_latent = v2v_ref_latent.to(device=device, dtype=dit_dtype)
                logger.info("V2V: using precached reference latent %s", v2v_ref_latent.shape)
                v2v_ref_path = None

        if v2v_ref_path:
            logger.info("V2V: encoding reference")
            try:
                vae_checkpoint = getattr(args, "vae", None) or getattr(args, "ltx2_checkpoint", None)
                if not vae_checkpoint:
                    raise ValueError("VAE checkpoint path required for V2V reference (--vae or --ltx2_checkpoint)")

                device = accelerator.device
                spatial_factor = 32
                width = sample_parameter.get("width", 768)
                height = sample_parameter.get("height", 512)
                width = (width // spatial_factor) * spatial_factor
                height = (height // spatial_factor) * spatial_factor

                ref_downscale = max(1, getattr(args, "reference_downscale", 1))
                if ref_downscale > 1:
                    ref_w = max((width // ref_downscale // spatial_factor) * spatial_factor, spatial_factor)
                    ref_h = max((height // ref_downscale // spatial_factor) * spatial_factor, spatial_factor)
                else:
                    ref_w, ref_h = width, height

                ref_frames = max(1, getattr(args, "reference_frames", 1))
                v2v_ref_latent = self._load_and_encode_v2v_reference(
                    ref_path=v2v_ref_path,
                    target_height=ref_h,
                    target_width=ref_w,
                    vae_checkpoint_path=vae_checkpoint,
                    device=device,
                    dtype=dit_dtype,
                    max_frames=ref_frames,
                )
            except Exception as e:
                logger.error(f"V2V: failed to load reference '{v2v_ref_path}': {e}")
                v2v_ref_latent = None

        ref_audio_latent = None
        ref_audio_path = sample_parameter.get("ref_audio_path") or sample_parameter.get("reference_audio_path")
        if "ref_audio_latent" in sample_parameter:
            ref_audio_latent = sample_parameter["ref_audio_latent"]
        elif "reference_audio_latent" in sample_parameter:
            ref_audio_latent = sample_parameter["reference_audio_latent"]

        if ref_audio_latent is not None:
            device = accelerator.device
            if isinstance(ref_audio_latent, torch.Tensor):
                if ref_audio_latent.dim() == 3:
                    ref_audio_latent = ref_audio_latent.unsqueeze(0)
                ref_audio_latent = ref_audio_latent.to(device=device, dtype=torch.float32)
                logger.info("Audio-ref: using precached reference audio latent %s", tuple(ref_audio_latent.shape))
                ref_audio_path = None
            else:
                logger.warning("Audio-ref: ignoring non-tensor ref_audio_latent of type %s", type(ref_audio_latent))
                ref_audio_latent = None

        if ref_audio_path and ref_audio_latent is None:
            logger.info("Audio-ref: encoding reference audio")
            try:
                checkpoint_path = getattr(args, "ltx2_checkpoint", None)
                if not checkpoint_path:
                    raise ValueError("--ltx2_checkpoint is required for reference-audio encoding")
                ref_audio_latent = self._load_and_encode_reference_audio_latent(
                    audio_path=ref_audio_path,
                    checkpoint_path=checkpoint_path,
                    device=accelerator.device,
                    dtype=dit_dtype,
                )
            except Exception as e:
                logger.error("Audio-ref: failed to load reference audio '%s': %s", ref_audio_path, e)
                ref_audio_latent = None

        lora_count = ensure_adapters_enabled_for_sampling(transformer)
        adapter_summary = summarize_active_adapters(transformer)
        if lora_count:
            logger.info("Sampling: LoRA modules active in transformer: %s", lora_count)
            if adapter_summary["lycoris"] > 0:
                logger.info(
                    "Sampling LyCORIS summary: active=%d blocks=%d attn1=%d attn2=%d ff=%d audio=%d quantized_origins=%d",
                    adapter_summary["lycoris"],
                    adapter_summary["block_count"],
                    adapter_summary["attn1"],
                    adapter_summary["attn2"],
                    adapter_summary["ff"],
                    adapter_summary["audio"],
                    adapter_summary["lycoris_quantized_origin"],
                )
            lora_stats = get_adapter_norm_samples(transformer)
            for stat in lora_stats:
                logger.info("Sampling LoRA norm: %s", stat)
        else:
            logger.warning("Sampling: no LoRA modules detected on transformer")

        loaded_vae = False
        if vae is None or getattr(vae, "_deferred", False):
            vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
            vae = self._load_vae_impl(args, vae_dtype=vae_dtype, vae_path=args.vae)
            loaded_vae = True

        # Use pre-loaded audio components if provided, otherwise load here (fallback for non-offload mode)
        loaded_audio = False
        disable_audio_preview = bool(getattr(args, "sample_disable_audio", False))
        use_audio_subprocess = bool(getattr(args, "sample_audio_subprocess", True))
        audio_only_preview = bool(getattr(args, "sample_audio_only", False))
        # When training mode is audio-only, inference must also use audio_only=True
        # to avoid context embedding split corruption and incorrect video modality.
        if self._ltx_mode == "audio":
            audio_only_preview = True
        if audio_only_preview and getattr(args, "ltx_mode", "video") not in {"av", "audio"}:
            raise ValueError("--sample_audio_only requires --ltx2_mode av or audio")
        enable_audio_preview = (self._audio_video or audio_only_preview) and not disable_audio_preview
        resolved_ic_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()
        audio_ref_only_sampling = (
            resolved_ic_strategy == "audio_ref_only_ic"
            and self._ltx_mode in {"av", "audio"}
            and isinstance(ref_audio_latent, torch.Tensor)
        )
        av_ic_sampling = (
            resolved_ic_strategy == "av_ic"
            and self._ltx_mode == "av"
            and ref_audio_latent is not None
            and v2v_ref_latent is not None
        )
        video_ref_only_av_sampling = (
            resolved_ic_strategy == "video_ref_only_av"
            and self._ltx_mode == "av"
            and v2v_ref_latent is not None
        )
        if isinstance(ref_audio_latent, torch.Tensor) and resolved_ic_strategy not in ("audio_ref_only_ic", "av_ic"):
            logger.warning(
                "Audio-ref latent provided but --ic_lora_strategy is '%s'; sample will ignore reference audio.",
                resolved_ic_strategy,
            )
            ref_audio_latent = None
            audio_ref_only_sampling = False
        force_audio_conditioning = audio_ref_only_sampling or av_ic_sampling or video_ref_only_av_sampling

        # Only load audio components here if NOT in offloading mode and not pre-loaded
        # In offloading mode with subprocess enabled (default), audio is decoded in a subprocess.
        # With --no-sample_audio_subprocess, audio is loaded lazily in-process during decode phase.
        sample_with_offloading = bool(getattr(args, "sample_with_offloading", False))
        if (
            audio_decoder is None
            and vocoder is None
            and enable_audio_preview
            and getattr(args, "ltx_mode", "video") in {"av", "audio"}
        ):
            if not sample_with_offloading and not use_audio_subprocess:
                # High VRAM mode without subprocess: load audio to GPU now (everything fits)
                audio_dtype = torch.bfloat16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
                try:
                    audio_decoder, vocoder = self._load_audio_components(
                        args,
                        audio_dtype=audio_dtype,
                        checkpoint_path=args.ltx2_checkpoint,
                        device=accelerator.device,
                    )
                    loaded_audio = True
                except Exception as exc:
                    logger.warning("Sampling audio decoder load failed; continuing without audio preview: %s", exc)
                    audio_decoder, vocoder = None, None
                    loaded_audio = False
            # else: subprocess mode or offloading mode - audio will be decoded later

        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 768)
        height = sample_parameter.get("height", 512)
        frame_count = sample_parameter.get("frame_count", 45)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 5.0)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)
        effective_cfg_scale = cfg_scale if cfg_scale is not None else guidance_scale
        video_cfg_scale_arg = getattr(args, "video_cfg_scale", None)
        audio_cfg_scale_arg = getattr(args, "audio_cfg_scale", None)
        do_classifier_free_guidance = (
            float(effective_cfg_scale) != 1.0
            or (video_cfg_scale_arg is not None and float(video_cfg_scale_arg) != 1.0)
            or (audio_cfg_scale_arg is not None and float(audio_cfg_scale_arg) != 1.0)
        )
        if do_classifier_free_guidance and negative_prompt is None:
            # CFG path still uses unconditional embedding (empty prompt).
            negative_prompt = ""
            sample_parameter["negative_prompt"] = negative_prompt

        spatial_factor = int(getattr(vae, "spatial_downsample_factor", 32))
        temporal_factor = int(getattr(vae, "temporal_downsample_factor", 8))
        width = (width // spatial_factor) * spatial_factor
        height = (height // spatial_factor) * spatial_factor
        frame_count = (frame_count - 1) // temporal_factor * temporal_factor + 1

        loaded_text_encoder = False
        strict_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        missing_prompt_embeds = sample_parameter.get("prompt_embeds") is None
        missing_negative_embeds = do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None
        if strict_precached and (missing_prompt_embeds or missing_negative_embeds):
            missing_parts = []
            if missing_prompt_embeds:
                missing_parts.append("prompt")
            if missing_negative_embeds:
                missing_parts.append("negative")
            missing_desc = "/".join(missing_parts)
            raise ValueError(
                "Precached sample prompt embeddings are incomplete; refusing to load Gemma during training. "
                f"Missing {missing_desc} embeddings for sample index {sample_parameter.get('enum', 0)}. "
                "Rebuild sample prompt cache with ltx2_cache_text_encoder_outputs.py."
            )

        if sample_parameter.get("prompt_embeds") is None:
            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            prompt_embeds, prompt_mask = self._encode_prompt_text(accelerator, prompt, text_encoder_dtype)
            sample_parameter["prompt_embeds"] = prompt_embeds
            sample_parameter["prompt_attention_mask"] = prompt_mask
            if do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None:
                neg_embeds, neg_mask = self._encode_prompt_text(
                    accelerator, negative_prompt, text_encoder_dtype
                )
                sample_parameter["negative_prompt_embeds"] = neg_embeds
                sample_parameter["negative_prompt_attention_mask"] = neg_mask
            loaded_text_encoder = True
        elif do_classifier_free_guidance and sample_parameter.get("negative_prompt_embeds") is None:
            text_encoder_dtype = self._build_text_encoder(args, accelerator)
            neg_embeds, neg_mask = self._encode_prompt_text(
                accelerator, negative_prompt, text_encoder_dtype
            )
            sample_parameter["negative_prompt_embeds"] = neg_embeds
            sample_parameter["negative_prompt_attention_mask"] = neg_mask
            loaded_text_encoder = True

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        # (I2V encoding now happens at the start of the method, before any model loading)

        do_classifier_free_guidance = (
            float(effective_cfg_scale) != 1.0
            or (video_cfg_scale_arg is not None and float(video_cfg_scale_arg) != 1.0)
            or (audio_cfg_scale_arg is not None and float(audio_cfg_scale_arg) != 1.0)
        )
        if do_classifier_free_guidance:
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")

        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )
        wav_path = os.path.join(save_dir, save_path) + ".wav"

        # Check if two-stage inference is enabled
        use_two_stage = bool(getattr(args, "sample_two_stage", False))
        spatial_upsampler_path = getattr(args, "spatial_upsampler_path", None)
        distilled_lora_path = getattr(args, "distilled_lora_path", None)
        enable_audio_conditioning = bool(enable_audio_preview) or bool(force_audio_conditioning)

        if use_two_stage:
            if not spatial_upsampler_path:
                logger.warning("Two-stage inference requested but --spatial_upsampler_path not set; falling back to single-stage")
                use_two_stage = False
            elif force_audio_conditioning:
                logger.warning(
                    "Reference-audio conditioning is not supported with two-stage inference; falling back to single-stage"
                )
                use_two_stage = False

        if use_two_stage:
            if v2v_ref_latent is not None:
                logger.warning("V2V reference conditioning is not supported with two-stage inference; ignoring V2V reference")
                v2v_ref_latent = None
            video, audio_waveform = self.do_inference_two_stage(
                accelerator=accelerator,
                args=args,
                sample_parameter=sample_parameter,
                vae=vae,
                dit_dtype=dit_dtype,
                transformer=transformer,
                width=width,
                height=height,
                frame_count=frame_count,
                sample_steps=sample_steps,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                seed=seed,
                generator=generator,
                spatial_upsampler_path=spatial_upsampler_path,
                conditioning_latent=conditioning_latent,
                distilled_lora_path=distilled_lora_path,
                stage2_steps=int(getattr(args, "sample_stage2_steps", 3)),
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                enable_audio_preview=enable_audio_preview,
                decode_video=not audio_only_preview,
                audio_only=audio_only_preview,
            )
        else:
            video, audio_waveform = self.do_inference(
                accelerator,
                args,
                sample_parameter,
                vae,
                dit_dtype,
                transformer,
                discrete_flow_shift,
                sample_steps,
                width,
                height,
                frame_count,
                generator,
                do_classifier_free_guidance,
                guidance_scale,
                cfg_scale,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                offload_transformer_for_decode=bool(getattr(args, "sample_with_offloading", False)),
                transformer_offload_device=torch.device("cpu"),
                restore_transformer_device=not (getattr(args, "sample_with_offloading", False) and accelerator.device.type == "cuda"),
                audio_output_path=wav_path if enable_audio_preview else None,
                use_audio_subprocess=use_audio_subprocess,
                enable_audio_preview=enable_audio_conditioning,
                decode_video=not audio_only_preview,
                audio_only=audio_only_preview,
                conditioning_latent=conditioning_latent,
                v2v_ref_latents=v2v_ref_latent,
                ref_audio_latents=ref_audio_latent,
            )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        if video is None and not audio_only_preview:
            logger.error("No video generated / 生成された動画がありません")
            return

        if getattr(args, "sample_include_reference", False) and video is not None:
            ref_path = sample_parameter.get("v2v_ref_path")
            if ref_path and os.path.exists(ref_path):
                try:
                    ref_video = self._load_reference_for_output(
                        ref_path, video.shape[3], video.shape[4], video.shape[2]
                    )
                    video = torch.cat([ref_video.to(video.device), video], dim=4)
                except Exception as e:
                    logger.warning(f"Failed to prepend reference to output: {e}")

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except:
            wandb = None

        video_path = None
        if video is not None:
            if video.shape[2] == 1:
                image_paths = save_images_grid(video, save_dir, save_path, create_subdir=False)
                if wandb_tracker is not None and wandb is not None:
                    for image_path in image_paths:
                        wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
            else:
                video_path = os.path.join(save_dir, save_path) + ".mp4"
                save_videos_grid(video, video_path)
                if wandb_tracker is not None and wandb is not None:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)
        if audio_waveform is not None:
            wav_path = os.path.join(save_dir, save_path) + ".wav"
            sample_rate = int(getattr(vocoder, "output_sample_rate", 24000)) if vocoder is not None else 24000
            self._save_audio_wav(wav_path, audio_waveform, sample_rate)
            if self._audio_metrics is not None:
                sample_metrics = self._audio_metrics.on_sample(
                    waveform=audio_waveform,
                    text_prompt=sample_parameter.get("prompt", ""),
                    device=device,
                    sample_rate=sample_rate,
                )
                if sample_metrics and len(accelerator.trackers) > 0:
                    accelerator.log(sample_metrics, step=steps)
            if getattr(args, "sample_merge_audio", False) and video_path is not None:
                merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
                self._mux_video_audio(video_path, wav_path, merged_path)
        elif getattr(args, "sample_merge_audio", False) and video_path is not None:
            wav_path = os.path.join(save_dir, save_path) + ".wav"
            if os.path.exists(wav_path):
                merged_path = os.path.join(save_dir, save_path) + "_av.mp4"
                self._mux_video_audio(video_path, wav_path, merged_path)

        if loaded_text_encoder:
            sample_parameter.pop("prompt_embeds", None)
            sample_parameter.pop("prompt_attention_mask", None)
            sample_parameter.pop("negative_prompt_embeds", None)
            sample_parameter.pop("negative_prompt_attention_mask", None)
            self._cleanup_text_encoder(accelerator)
        if loaded_vae:
            vae.to_device("cpu")
            clean_memory_on_device(device)
        if loaded_audio:
            audio_decoder.to("cpu")
            vocoder.to("cpu")
            clean_memory_on_device(device)

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict,
        vae,
        dit_dtype: torch.dtype,
        transformer,
        discrete_flow_shift: float,
        sample_steps: int,
        width: int,
        height: int,
        frame_count: int,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        image_path: Optional[str] = None,
        control_video_path: Optional[str] = None,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
        offload_transformer_for_decode: bool = False,
        transformer_offload_device: Optional[torch.device] = None,
        restore_transformer_device: bool = True,
        audio_output_path: Optional[str] = None,
        use_audio_subprocess: bool = False,
        enable_audio_preview: bool = False,
        decode_video: bool = True,
        audio_only: bool = False,
        conditioning_latent: Optional[torch.Tensor] = None,
        v2v_ref_latents: Optional[torch.Tensor] = None,
        ref_audio_latents: Optional[torch.Tensor] = None,
    ):
        """Generate sample video during training using LTX-2 denoising loop"""
        from musubi_tuner.ltx_2.types import AudioLatentShape, VideoPixelShape

        transformer_device = next(transformer.parameters()).device
        transformer_offload_device = transformer_offload_device or torch.device("cpu")
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)
        # Keep VAE off GPU during denoise when offloading is enabled.
        if not offload_transformer_for_decode:
            vae.to_device(transformer_device)
        vae.to_dtype(original_vae_dtype)

        # Get text embeddings
        prompt_embeds = sample_parameter.get("prompt_embeds")
        if prompt_embeds is None:
            raise ValueError("Sample parameter missing prompt embeddings")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=dit_dtype)

        prompt_mask = sample_parameter.get("prompt_attention_mask")
        def _normalize_prompt_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.dim() == 1:
                return mask.unsqueeze(0)
            if mask.dim() > 2:
                return mask.view(mask.shape[0], -1)
            return mask

        if do_classifier_free_guidance:
            negative_prompt_embeds = sample_parameter.get("negative_prompt_embeds")
            negative_prompt_mask = sample_parameter.get("negative_prompt_attention_mask")
            if negative_prompt_embeds is not None:
                if negative_prompt_embeds.dim() == 2:
                    negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=transformer_device, dtype=dit_dtype
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_mask = _normalize_prompt_mask(prompt_mask)
                negative_prompt_mask = _normalize_prompt_mask(negative_prompt_mask)
                if prompt_mask is not None and negative_prompt_mask is not None:
                    prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
                elif prompt_mask is not None:
                    logger.warning(
                        "Sampling: negative prompt mask missing; duplicating prompt mask."
                    )
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
            else:
                logger.warning(
                    "Sampling: negative prompt embeddings missing; duplicating prompt embeds."
                )
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                prompt_mask = _normalize_prompt_mask(prompt_mask)
                if prompt_mask is not None:
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
        if prompt_mask is not None:
            prompt_mask = _normalize_prompt_mask(prompt_mask)
        if prompt_mask is not None:
            mask_len = prompt_mask.shape[-1]
            embed_len = prompt_embeds.shape[1]
            if mask_len != embed_len:
                logger.warning(
                    "Sample prompt mask length %s != embeds length %s; aligning mask for sampling.",
                    mask_len,
                    embed_len,
                )
                if mask_len > embed_len:
                    # padding_side="left" in the Gemma encoder, keep rightmost tokens.
                    prompt_mask = prompt_mask[:, -embed_len:]
                else:
                    pad = embed_len - mask_len
                    prompt_mask = F.pad(prompt_mask, (pad, 0), value=1)
            if prompt_mask.shape[-1] != prompt_embeds.shape[1]:
                logger.warning(
                    "Sample prompt mask still mismatched after alignment (mask=%s, embeds=%s); disabling mask for sampling.",
                    prompt_mask.shape[-1],
                    prompt_embeds.shape[1],
                )
                prompt_mask = None
        prompt_mask = prompt_mask.to(device=transformer_device, dtype=torch.int64) if prompt_mask is not None else None

        resolved_ic_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()

        if ref_audio_latents is not None:
            if not isinstance(ref_audio_latents, torch.Tensor):
                raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
            if ref_audio_latents.dim() == 3:
                ref_audio_latents = ref_audio_latents.unsqueeze(0)
            if ref_audio_latents.dim() != 4:
                raise ValueError(
                    f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                )

        if ref_audio_latents is not None and resolved_ic_strategy not in ("audio_ref_only_ic", "av_ic"):
            logger.warning(
                "Sampling: reference-audio latents provided but --ic_lora_strategy is '%s'; ignoring ref-audio conditioning.",
                resolved_ic_strategy,
            )
            ref_audio_latents = None

        audio_ref_only_ic_sampling = (
            resolved_ic_strategy == "audio_ref_only_ic"
            and self._ltx_mode in {"av", "audio"}
            and ref_audio_latents is not None
        )
        av_ic_sampling = (
            resolved_ic_strategy == "av_ic"
            and self._ltx_mode == "av"
            and ref_audio_latents is not None
            and v2v_ref_latents is not None
        )
        video_ref_only_av_sampling = (
            resolved_ic_strategy == "video_ref_only_av"
            and self._ltx_mode == "av"
            and v2v_ref_latents is not None
        )

        attention_overrides = []
        if getattr(args, "sample_disable_flash_attn", True):
            from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction

            logger.info("Sampling: disabling FlashAttention for preview")
            attention_overrides = self._override_attention_function(
                transformer, AttentionFunction.PYTORCH
            )
            if prompt_mask is not None:
                logger.info("Sampling: disabling prompt attention mask for preview")
                prompt_mask = None

        enable_audio_preview = bool(enable_audio_preview)
        if not enable_audio_preview and not audio_ref_only_ic_sampling and not av_ic_sampling and not video_ref_only_av_sampling:
            expected_embed_dim = None
            try:
                caption_proj = getattr(transformer, "caption_projection", None)
                if caption_proj is not None and hasattr(caption_proj, "linear_1"):
                    expected_embed_dim = int(caption_proj.linear_1.in_features)
            except Exception:
                expected_embed_dim = None

            current_dim = int(prompt_embeds.shape[-1])
            if expected_embed_dim is not None and current_dim == expected_embed_dim * 2:
                logger.warning(
                    "Sampling: audio preview disabled; using video-only prompt embeddings (half of dim=%s).",
                    current_dim,
                )
                prompt_embeds = prompt_embeds[..., : expected_embed_dim]

        # Setup LTX-2 specific stepper
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler

        stepper = EulerDiffusionStep()

        # Calculate latent dimensions
        vae_scale_factor_temporal = getattr(vae, "temporal_downsample_factor", 4)
        vae_scale_factor_spatial = getattr(vae, "spatial_downsample_factor", 8)
        latent_frames = (frame_count - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        in_channels = getattr(transformer, "in_channels", 128)

        # Initialize latents
        latents = torch.randn(
            (1, int(in_channels), latent_frames, latent_height, latent_width),
            dtype=torch.float32,
            device=transformer_device,
            generator=generator,
        )

        # ===== AV_IC: combined video+audio IC-LoRA sampling path =====
        if av_ic_sampling and v2v_ref_latents is not None and isinstance(ref_audio_latents, torch.Tensor):
            video, audio_waveform = self._do_av_ic_denoising(
                latents=latents,
                v2v_ref_latents=v2v_ref_latents,
                ref_audio_latents=ref_audio_latents,
                transformer=transformer,
                dit_dtype=dit_dtype,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                sample_parameter=sample_parameter,
                sample_steps=sample_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                vae=vae,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                args=args,
                offload_transformer_for_decode=offload_transformer_for_decode,
                transformer_offload_device=transformer_offload_device,
                restore_transformer_device=restore_transformer_device,
                decode_video=decode_video,
                attention_overrides=attention_overrides,
            )
            return video, audio_waveform

        if video_ref_only_av_sampling and v2v_ref_latents is not None:
            video, audio_waveform = self._do_av_ic_denoising(
                latents=latents,
                v2v_ref_latents=v2v_ref_latents,
                ref_audio_latents=None,
                transformer=transformer,
                dit_dtype=dit_dtype,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                sample_parameter=sample_parameter,
                sample_steps=sample_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                vae=vae,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                args=args,
                offload_transformer_for_decode=offload_transformer_for_decode,
                transformer_offload_device=transformer_offload_device,
                restore_transformer_device=restore_transformer_device,
                decode_video=decode_video,
                attention_overrides=attention_overrides,
            )
            return video, audio_waveform

        # ===== V2V / IC-LoRA sampling path =====
        # Mirrors the training forward pass: patchify ref+target, build Modality with
        # per-token timesteps (ref=0, target=sigma), call base_model directly.
        if v2v_ref_latents is not None:
            video, audio_waveform = self._do_v2v_denoising(
                latents=latents,
                v2v_ref_latents=v2v_ref_latents,
                transformer=transformer,
                dit_dtype=dit_dtype,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                sample_parameter=sample_parameter,
                sample_steps=sample_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guidance_scale=guidance_scale,
                cfg_scale=cfg_scale,
                vae=vae,
                args=args,
                offload_transformer_for_decode=offload_transformer_for_decode,
                transformer_offload_device=transformer_offload_device,
                restore_transformer_device=restore_transformer_device,
                decode_video=decode_video,
                attention_overrides=attention_overrides,
            )
            return video, audio_waveform

        # Setup I2V conditioning mask if provided
        denoise_mask = None
        clean_latent = None
        i2v_conditioning_mask_tokens = None
        use_i2v_token_timestep_mask = bool(getattr(args, "sample_i2v_token_timestep_mask", True))
        if conditioning_latent is not None:
            # Validate conditioning_latent shape
            if conditioning_latent.dim() != 5:
                logger.warning(f"I2V: conditioning_latent has wrong dimensions {conditioning_latent.shape}, expected [B,C,T,H,W]. Skipping I2V conditioning.")
            elif conditioning_latent.shape[2] != 1:
                logger.warning(f"I2V: conditioning_latent has {conditioning_latent.shape[2]} frames, expected 1. Skipping I2V conditioning.")
            elif latents.shape[2] < 1:
                logger.warning("I2V: Video latents have no temporal frames. Skipping I2V conditioning.")
            elif conditioning_latent.shape[1] != latents.shape[1]:
                logger.warning(f"I2V: Channel dimension mismatch - conditioning {conditioning_latent.shape[1]} vs latents {latents.shape[1]}. Skipping I2V conditioning.")
            elif conditioning_latent.shape[-2:] != latents.shape[-2:]:
                logger.warning(f"I2V: Spatial dimension mismatch - conditioning {conditioning_latent.shape[-2:]} vs latents {latents.shape[-2:]}. Skipping I2V conditioning.")
            else:
                try:
                    cond_on_device = conditioning_latent.to(device=latents.device, dtype=latents.dtype)

                    # CRITICAL: Initialize first frame of latents with conditioning image
                    # This ensures the first frame starts as the conditioning, not random noise
                    latents[:, :, 0:1, :, :] = cond_on_device

                    # Create denoise_mask: 0.0 for first frame (locked), 1.0 for others (denoised)
                    denoise_mask = torch.ones_like(latents)
                    denoise_mask[:, :, 0:1, :, :] = 0.0

                    # Store clean conditioning latent (will be blended back at each step)
                    clean_latent = torch.zeros_like(latents)
                    clean_latent[:, :, 0:1, :, :] = cond_on_device

                    if use_i2v_token_timestep_mask:
                        bsz, _c, frames, h_lat, w_lat = latents.shape
                        seq_len = frames * h_lat * w_lat
                        first_frame_tokens = h_lat * w_lat
                        i2v_conditioning_mask_tokens = torch.zeros(
                            (bsz, seq_len),
                            device=latents.device,
                            dtype=torch.bool,
                        )
                        if first_frame_tokens > 0:
                            i2v_conditioning_mask_tokens[:, :first_frame_tokens] = True
                        logger.info("I2V: enabled token timestep mask for conditioned first-frame tokens")

                    logger.info(f"I2V: Initialized first frame conditioning (shape: {conditioning_latent.shape})")
                except Exception as e:
                    logger.error(f"I2V: Failed to setup conditioning: {e}", exc_info=True)
                    denoise_mask = None
                    clean_latent = None
                    i2v_conditioning_mask_tokens = None

        # Setup scheduler without passing latent; this uses default MAX_SHIFT_ANCHOR=4096.
        ltx2_scheduler = LTX2Scheduler()
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)

        audio_latents = None
        ref_audio_latents_device = None
        ref_audio_seq_len = 0
        if enable_audio_preview or audio_ref_only_ic_sampling or video_ref_only_av_sampling:
            frame_rate = sample_parameter.get("frame_rate", 25)
            video_shape = VideoPixelShape(
                batch=1,
                frames=int(frame_count),
                height=int(height),
                width=int(width),
                fps=float(frame_rate),
            )
            audio_cfg = self._get_audio_preview_config(args, transformer)
            channels = int(audio_cfg["channels"])
            mel_bins = int(audio_cfg["mel_bins"])
            sample_rate = int(audio_cfg["sample_rate"])
            hop_length = int(audio_cfg["hop_length"])
            audio_downsample = int(audio_cfg["audio_latent_downsample_factor"])
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                video_shape,
                channels=channels,
                mel_bins=mel_bins,
                sample_rate=sample_rate,
                hop_length=hop_length,
                audio_latent_downsample_factor=audio_downsample,
            )
            audio_frames = max(int(audio_shape.frames), 1)
            audio_latents = torch.randn(
                (1, channels, audio_frames, mel_bins),
                dtype=torch.float32,
                device=transformer_device,
                generator=generator,
            )

            if audio_ref_only_ic_sampling:
                if ref_audio_latents is None:
                    raise ValueError("audio_ref_only_ic sampling requires reference-audio latents")
                ref_audio_latents_device = ref_audio_latents.to(device=transformer_device, dtype=torch.float32)
                if int(ref_audio_latents_device.shape[0]) != int(audio_latents.shape[0]):
                    raise ValueError(
                        f"Batch mismatch for reference-audio: ref={tuple(ref_audio_latents_device.shape)} target={tuple(audio_latents.shape)}"
                    )
                if int(ref_audio_latents_device.shape[1]) != channels or int(ref_audio_latents_device.shape[3]) != mel_bins:
                    raise ValueError(
                        "Reference-audio latent channel/mel mismatch. "
                        f"Got ref={tuple(ref_audio_latents_device.shape)} target={tuple(audio_latents.shape)}"
                    )
                ref_audio_seq_len = int(ref_audio_latents_device.shape[2])
                if ref_audio_seq_len <= 0:
                    raise ValueError("Reference-audio latent sequence length must be > 0")

        # Identity guidance setup
        _identity_guidance_scale = 0.0
        if audio_ref_only_ic_sampling and ref_audio_seq_len > 0:
            _identity_guidance_scale = float(getattr(args, "audio_ref_identity_guidance_scale", 0.0) or 0.0)
            if _identity_guidance_scale > 0.0:
                logger.info(
                    "Sampling: identity guidance scale=%.2f (extra forward pass per step without reference)",
                    _identity_guidance_scale,
                )

        # AV bimodal CFG setup
        _video_cfg_scale = float(
            getattr(args, "video_cfg_scale", None)
            if getattr(args, "video_cfg_scale", None) is not None
            else (cfg_scale if cfg_scale is not None else guidance_scale)
        )
        _audio_cfg_scale = float(
            getattr(args, "audio_cfg_scale", None)
            if getattr(args, "audio_cfg_scale", None) is not None
            else (cfg_scale if cfg_scale is not None else guidance_scale)
        )
        _video_modality_scale = float(
            getattr(args, "video_modality_scale", None)
            if getattr(args, "video_modality_scale", None) is not None
            else getattr(args, "av_bimodal_scale", 1.0)
        )
        _audio_modality_scale = float(
            getattr(args, "audio_modality_scale", None)
            if getattr(args, "audio_modality_scale", None) is not None
            else getattr(args, "av_bimodal_scale", 1.0)
        )
        _video_rescale_scale = float(
            getattr(args, "video_rescale_scale", None)
            if getattr(args, "video_rescale_scale", None) is not None
            else getattr(args, "rescale_scale", 0.0)
        )
        _audio_rescale_scale = float(
            getattr(args, "audio_rescale_scale", None)
            if getattr(args, "audio_rescale_scale", None) is not None
            else getattr(args, "rescale_scale", 0.0)
        )
        _av_bimodal_cfg = bool(getattr(args, "av_bimodal_cfg", False)) or (
            audio_latents is not None and (_video_modality_scale != 1.0 or _audio_modality_scale != 1.0)
        )
        if _av_bimodal_cfg and audio_latents is not None:
            from musubi_tuner.ltx_2.guidance.perturbations import (
                BatchedPerturbationConfig as _BPC,
                Perturbation as _Pert,
                PerturbationConfig as _PertCfg,
                PerturbationType as _PertType,
            )
            _bimodal_pert_single = _PertCfg(perturbations=[
                _Pert(type=_PertType.SKIP_A2V_CROSS_ATTN, blocks=None),
                _Pert(type=_PertType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ])
            logger.info(
                "Sampling: modality guidance video=%.2f audio=%.2f (extra forward pass without cross-modal attention)",
                _video_modality_scale,
                _audio_modality_scale,
            )
        else:
            _av_bimodal_cfg = False

        # Denoising loop using LTX-2 scheduler with sigmas
        with torch.no_grad():
            for step_idx in tqdm(range(len(sigmas) - 1), desc="LTX-2 preview", leave=False):
                sigma = sigmas[step_idx]

                # Expand for CFG if needed
                latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype=dit_dtype)

                audio_model_input = None
                audio_timestep_for_model = None
                if audio_latents is not None:
                    if audio_ref_only_ic_sampling and ref_audio_latents_device is not None and ref_audio_seq_len > 0:
                        combined_audio = torch.cat([ref_audio_latents_device, audio_latents], dim=2)
                        audio_model_input = (
                            torch.cat([combined_audio, combined_audio], dim=0)
                            if do_classifier_free_guidance
                            else combined_audio
                        )
                        audio_model_input = audio_model_input.to(dtype=dit_dtype)

                        tgt_seq_len = int(audio_latents.shape[2])
                        target_audio_timestep = sigma.expand(tgt_seq_len).view(1, -1).to(
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                        ref_audio_timestep = torch.zeros(
                            (1, ref_audio_seq_len),
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                        audio_timestep_for_model = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)
                        if do_classifier_free_guidance:
                            audio_timestep_for_model = audio_timestep_for_model.repeat(2, 1)
                    else:
                        audio_model_input = (
                            torch.cat([audio_latents, audio_latents], dim=0)
                            if do_classifier_free_guidance
                            else audio_latents
                        )
                        audio_model_input = audio_model_input.to(dtype=dit_dtype)
                        audio_timestep_for_model = sigma.expand(audio_model_input.shape[0]).to(
                            device=transformer_device,
                            dtype=dit_dtype,
                        ).unsqueeze(1)

                # Prepare timestep (sigma in [0, 1])
                timestep_for_model = sigma.expand(latent_model_input.shape[0]).to(device=transformer_device, dtype=dit_dtype)

                resolved_transformer_options = {"patches_replace": {}}
                if i2v_conditioning_mask_tokens is not None:
                    video_conditioning_mask_tokens = i2v_conditioning_mask_tokens
                    if do_classifier_free_guidance:
                        video_conditioning_mask_tokens = torch.cat(
                            [video_conditioning_mask_tokens, video_conditioning_mask_tokens],
                            dim=0,
                        )
                    resolved_transformer_options["video_conditioning_mask"] = video_conditioning_mask_tokens

                if (
                    audio_ref_only_ic_sampling
                    and audio_model_input is not None
                    and ref_audio_seq_len > 0
                ):
                    # ID-LoRA disables attention masks during inference (validation config
                    # sets mask_cross_attention_to_reference=false, mask_ref_audio_to_text=false).
                    # The masks are training scaffolding: they force the model to learn proper
                    # attention patterns, but at inference time the LoRA weights have already
                    # internalized the separation.  Only position overrides are kept.
                    sampling_args = copy.copy(args)
                    sampling_args.audio_ref_mask_cross_attention_to_reference = False
                    sampling_args.audio_ref_mask_reference_from_text_attention = False
                    resolved_transformer_options.update(
                        self._build_audio_ref_transformer_overrides(
                            args=sampling_args,
                            transformer=transformer,
                            video_latents=latent_model_input,
                            text_embeds=prompt_embeds,
                            text_mask=prompt_mask,
                            audio_model_latents=audio_model_input,
                            ref_audio_seq_len=ref_audio_seq_len,
                            device=transformer_device,
                            dtype=dit_dtype,
                        )
                    )

                # Model prediction
                if self._audio_video and audio_model_input is not None:
                    model_input = [latent_model_input, audio_model_input]
                else:
                    model_input = latent_model_input

                model_pred = transformer(
                    model_input,
                    timestep=timestep_for_model.unsqueeze(1),  # [B, 1] for per-token timesteps
                    audio_timestep=audio_timestep_for_model,
                    context=prompt_embeds,
                    attention_mask=prompt_mask,
                    frame_rate=sample_parameter.get("frame_rate", 25),
                    transformer_options=resolved_transformer_options,
                    audio_only=audio_only,
                )

                audio_pred = None
                if isinstance(model_pred, (list, tuple)):
                    video_pred, audio_pred = model_pred
                else:
                    video_pred = model_pred

                if audio_ref_only_ic_sampling and audio_pred is not None and ref_audio_seq_len > 0:
                    if int(audio_pred.shape[2]) <= ref_audio_seq_len:
                        raise ValueError(
                            f"audio_pred length {audio_pred.shape[2]} is too short for ref_audio_seq_len={ref_audio_seq_len}"
                        )
                    audio_pred = audio_pred[:, :, ref_audio_seq_len:, :]

                # IMPORTANT: Convert velocity to x0 FIRST, then apply CFG to x0
                # X0Model wraps the velocity model before guidance is applied.
                # and CFG is applied to denoised (x0) outputs, not velocity predictions
                video_pred = video_pred.to(dtype=latents.dtype)

                sigma_for_video = denoise_mask * sigma if denoise_mask is not None else sigma

                # --- Video CFG ---
                x0_cond = None
                if do_classifier_free_guidance:
                    vel_uncond, vel_cond = video_pred.chunk(2)
                    x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond, sigma_for_video)
                    x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond, sigma_for_video)
                    video_x0 = x0_uncond + _video_cfg_scale * (x0_cond - x0_uncond)
                else:
                    video_x0 = X0PredictionWrapper.velocity_to_x0(latents, video_pred, sigma_for_video)

                # --- Audio CFG ---
                audio_x0 = None
                aud_x0_cond = None
                if audio_pred is not None and audio_latents is not None:
                    audio_pred = audio_pred.to(dtype=audio_latents.dtype)
                    if do_classifier_free_guidance:
                        aud_vel_uncond, aud_vel_cond = audio_pred.chunk(2)
                        aud_x0_uncond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_uncond, sigma.item())
                        aud_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_cond, sigma.item())
                        audio_x0 = aud_x0_uncond + _audio_cfg_scale * (aud_x0_cond - aud_x0_uncond)
                    else:
                        audio_x0 = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_pred, sigma.item())

                    # --- Identity guidance: extra forward pass without reference audio ---
                    # Isolates the reference audio's contribution and amplifies it (post-CFG).
                    # Formula: audio_x0 += scale * (cond_with_ref - cond_without_ref)
                    if _identity_guidance_scale > 0.0 and aud_x0_cond is not None:
                        noref_audio = audio_latents.to(dtype=dit_dtype)
                        tgt_seq_len = int(audio_latents.shape[2])
                        noref_audio_ts = sigma.expand(tgt_seq_len).view(1, -1).to(
                            device=transformer_device, dtype=dit_dtype,
                        )
                        noref_video = latents.to(dtype=dit_dtype)
                        noref_video_ts = sigma.expand(1).to(device=transformer_device, dtype=dit_dtype)

                        cond_prompt = prompt_embeds[1:2]  # conditional half only
                        cond_mask = prompt_mask[1:2] if prompt_mask is not None else None

                        noref_options = {"patches_replace": {}}
                        if i2v_conditioning_mask_tokens is not None:
                            noref_options["video_conditioning_mask"] = i2v_conditioning_mask_tokens

                        noref_input = [noref_video, noref_audio] if self._audio_video else noref_video
                        noref_pred = transformer(
                            noref_input,
                            timestep=noref_video_ts.unsqueeze(1),
                            audio_timestep=noref_audio_ts,
                            context=cond_prompt,
                            attention_mask=cond_mask,
                            frame_rate=sample_parameter.get("frame_rate", 25),
                            transformer_options=noref_options,
                            audio_only=audio_only,
                        )

                        noref_audio_vel = noref_pred[1] if isinstance(noref_pred, (list, tuple)) else None
                        if noref_audio_vel is not None:
                            noref_audio_vel = noref_audio_vel.to(dtype=audio_latents.dtype)
                            aud_x0_noref = X0PredictionWrapper.velocity_to_x0(
                                audio_latents, noref_audio_vel, sigma.item()
                            )
                            audio_x0 = audio_x0 + _identity_guidance_scale * (aud_x0_cond - aud_x0_noref)

                # --- AV Bimodal CFG: extra forward pass with cross-modal attention disabled ---
                # Strengthens independent modality generation by contrasting full cross-attention
                # prediction with one where A2V and V2A attention are skipped.
                # Formula: x0 += (scale - 1) * (cond_full - cond_bimodal)
                if _av_bimodal_cfg and do_classifier_free_guidance and audio_pred is not None:
                    # Single-batch (conditional only) — we only need the cond prediction.
                    bimodal_perturbations = _BPC(perturbations=[_bimodal_pert_single])

                    bimodal_options = {"patches_replace": {}, "perturbations": bimodal_perturbations}
                    if i2v_conditioning_mask_tokens is not None:
                        bimodal_options["video_conditioning_mask"] = i2v_conditioning_mask_tokens
                    if (
                        audio_ref_only_ic_sampling
                        and audio_model_input is not None
                        and ref_audio_seq_len > 0
                    ):
                        bm_sampling_args = copy.copy(args)
                        bm_sampling_args.audio_ref_mask_cross_attention_to_reference = False
                        bm_sampling_args.audio_ref_mask_reference_from_text_attention = False
                        bimodal_options.update(
                            self._build_audio_ref_transformer_overrides(
                                args=bm_sampling_args,
                                transformer=transformer,
                                video_latents=latents.to(dtype=dit_dtype).unsqueeze(0) if latents.dim() == 4 else latents.to(dtype=dit_dtype),
                                text_embeds=prompt_embeds[1:2],
                                text_mask=prompt_mask[1:2] if prompt_mask is not None else None,
                                audio_model_latents=audio_model_input[:1] if audio_model_input is not None else None,
                                ref_audio_seq_len=ref_audio_seq_len,
                                device=transformer_device,
                                dtype=dit_dtype,
                            )
                        )

                    # Conditional-only video and audio inputs (single batch)
                    bm_video = latents.to(dtype=dit_dtype)
                    bm_video_ts = sigma.expand(1).to(device=transformer_device, dtype=dit_dtype)

                    if audio_ref_only_ic_sampling and ref_audio_latents_device is not None and ref_audio_seq_len > 0:
                        bm_audio_input = torch.cat([ref_audio_latents_device, audio_latents], dim=2).to(dtype=dit_dtype)
                        tgt_seq = int(audio_latents.shape[2])
                        bm_tgt_ts = sigma.expand(tgt_seq).view(1, -1).to(device=transformer_device, dtype=dit_dtype)
                        bm_ref_ts = torch.zeros((1, ref_audio_seq_len), device=transformer_device, dtype=dit_dtype)
                        bm_audio_ts = torch.cat([bm_ref_ts, bm_tgt_ts], dim=1)
                    elif audio_latents is not None:
                        bm_audio_input = audio_latents.to(dtype=dit_dtype)
                        bm_audio_ts = sigma.expand(int(audio_latents.shape[2])).view(1, -1).to(
                            device=transformer_device, dtype=dit_dtype,
                        )
                    else:
                        bm_audio_input = None
                        bm_audio_ts = None

                    cond_prompt = prompt_embeds[1:2]
                    cond_mask = prompt_mask[1:2] if prompt_mask is not None else None

                    bm_input = [bm_video, bm_audio_input] if self._audio_video and bm_audio_input is not None else bm_video

                    bimodal_pred = transformer(
                        bm_input,
                        timestep=bm_video_ts.unsqueeze(1),
                        audio_timestep=bm_audio_ts,
                        context=cond_prompt,
                        attention_mask=cond_mask,
                        frame_rate=sample_parameter.get("frame_rate", 25),
                        transformer_options=bimodal_options,
                        audio_only=audio_only,
                    )

                    if isinstance(bimodal_pred, (list, tuple)):
                        bm_video_pred, bm_audio_pred = bimodal_pred
                    else:
                        bm_video_pred, bm_audio_pred = bimodal_pred, None

                    # Bimodal delta for video
                    if bm_video_pred is not None and x0_cond is not None:
                        bm_video_pred = bm_video_pred.to(dtype=latents.dtype)
                        bm_x0_cond = X0PredictionWrapper.velocity_to_x0(latents, bm_video_pred, sigma_for_video)
                        video_x0 = video_x0 + (_video_modality_scale - 1) * (x0_cond - bm_x0_cond)

                    # Bimodal delta for audio
                    if bm_audio_pred is not None and aud_x0_cond is not None:
                        if audio_ref_only_ic_sampling and ref_audio_seq_len > 0:
                            bm_audio_pred = bm_audio_pred[:, :, ref_audio_seq_len:, :]
                        bm_audio_pred = bm_audio_pred.to(dtype=audio_latents.dtype)
                        bm_aud_x0_cond = X0PredictionWrapper.velocity_to_x0(
                            audio_latents, bm_audio_pred, sigma.item()
                        )
                        audio_x0 = audio_x0 + (_audio_modality_scale - 1) * (aud_x0_cond - bm_aud_x0_cond)

                # --- STG (Spatio-Temporal Guidance): perturbed forward + steer x0 ---
                # One extra forward with self-attention skipped at --stg_blocks; video_x0
                # is pushed by stg_scale * (x0_cond - x0_perturbed). Opt-in; inert at 0.
                _stg_scale = float(getattr(args, "stg_scale", 0.0) or 0.0)
                if _stg_scale > 0.0 and video_pred is not None:
                    from musubi_tuner.ltx_2.guidance.perturbations import (
                        BatchedPerturbationConfig as _BPC_STG,
                        Perturbation as _Pert_STG,
                        PerturbationConfig as _PertCfg_STG,
                        PerturbationType as _PertType_STG,
                    )
                    _stg_blocks_arg = getattr(args, "stg_blocks", None)
                    _stg_mode_arg = str(getattr(args, "stg_mode", "video"))
                    _stg_audio_allowed = (
                        _stg_mode_arg in ("audio", "both")
                        and audio_pred is not None
                        and audio_latents is not None
                        and not audio_ref_only_ic_sampling
                    )

                    _stg_pert_list = []
                    if _stg_mode_arg in ("video", "both"):
                        _stg_pert_list.append(
                            _Pert_STG(type=_PertType_STG.SKIP_VIDEO_SELF_ATTN, blocks=_stg_blocks_arg)
                        )
                    if _stg_audio_allowed:
                        _stg_pert_list.append(
                            _Pert_STG(type=_PertType_STG.SKIP_AUDIO_SELF_ATTN, blocks=_stg_blocks_arg)
                        )

                    if _stg_pert_list:
                        _stg_perturbations = _BPC_STG(
                            perturbations=[_PertCfg_STG(perturbations=_stg_pert_list)]
                        )
                        _stg_options = {"patches_replace": {}, "perturbations": _stg_perturbations}
                        if i2v_conditioning_mask_tokens is not None:
                            _stg_options["video_conditioning_mask"] = i2v_conditioning_mask_tokens

                        if do_classifier_free_guidance:
                            _half = prompt_embeds.shape[0] // 2
                            _stg_ctx = prompt_embeds[_half:]
                            _stg_mask = prompt_mask[_half:] if prompt_mask is not None else None
                        else:
                            _stg_ctx = prompt_embeds
                            _stg_mask = prompt_mask

                        _stg_video = latents.to(dtype=dit_dtype)
                        _stg_video_ts = sigma.expand(1).to(device=transformer_device, dtype=dit_dtype)

                        _stg_audio = None
                        _stg_audio_ts = None
                        if _stg_audio_allowed:
                            _stg_audio = audio_latents.to(dtype=dit_dtype)
                            _stg_audio_ts = sigma.expand(int(audio_latents.shape[2])).view(1, -1).to(
                                device=transformer_device, dtype=dit_dtype,
                            )

                        _stg_input = (
                            [_stg_video, _stg_audio]
                            if self._audio_video and _stg_audio is not None
                            else _stg_video
                        )

                        _stg_pred = transformer(
                            _stg_input,
                            timestep=_stg_video_ts.unsqueeze(1),
                            audio_timestep=_stg_audio_ts,
                            context=_stg_ctx,
                            attention_mask=_stg_mask,
                            frame_rate=sample_parameter.get("frame_rate", 25),
                            transformer_options=_stg_options,
                            audio_only=audio_only,
                        )

                        if isinstance(_stg_pred, (list, tuple)):
                            _stg_vpred, _stg_apred = _stg_pred
                        else:
                            _stg_vpred, _stg_apred = _stg_pred, None

                        _v_base = x0_cond if x0_cond is not None else video_x0
                        if _stg_mode_arg in ("video", "both") and _stg_vpred is not None:
                            _stg_vpred = _stg_vpred.to(dtype=latents.dtype)
                            _v_ptb = X0PredictionWrapper.velocity_to_x0(latents, _stg_vpred, sigma_for_video)
                            video_x0 = video_x0 + _stg_scale * (_v_base - _v_ptb)

                        if _stg_audio_allowed and _stg_apred is not None and audio_x0 is not None:
                            _a_base = aud_x0_cond if aud_x0_cond is not None else audio_x0
                            _stg_apred = _stg_apred.to(dtype=audio_latents.dtype)
                            _a_ptb = X0PredictionWrapper.velocity_to_x0(audio_latents, _stg_apred, sigma.item())
                            audio_x0 = audio_x0 + _stg_scale * (_a_base - _a_ptb)

                # --- CFG\u2605 rescaling. 0.0 disables. ---
                if _video_rescale_scale > 0.0:
                    if x0_cond is not None and video_x0 is not None:
                        _ps = video_x0.std()
                        if _ps > 1e-6:
                            _f = x0_cond.std() / _ps
                            _f = _video_rescale_scale * _f + (1.0 - _video_rescale_scale)
                            video_x0 = video_x0 * _f
                if _audio_rescale_scale > 0.0:
                    if aud_x0_cond is not None and audio_x0 is not None:
                        _ps = audio_x0.std()
                        if _ps > 1e-6:
                            _f = aud_x0_cond.std() / _ps
                            _f = _audio_rescale_scale * _f + (1.0 - _audio_rescale_scale)
                            audio_x0 = audio_x0 * _f

                # --- Video: denoise mask blend + Euler step + hard-lock ---
                if denoise_mask is not None and clean_latent is not None:
                    video_x0 = video_x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)
                latents = stepper.step(latents, video_x0, sigmas, step_idx)
                if denoise_mask is not None and clean_latent is not None:
                    latents = latents * denoise_mask + clean_latent * (1.0 - denoise_mask)

                # --- Audio: Euler step ---
                if audio_x0 is not None and audio_latents is not None:
                    audio_latents = stepper.step(audio_latents, audio_x0, sigmas, step_idx)

        # Free I2V conditioning tensors to reclaim memory before VAE decode
        if denoise_mask is not None or clean_latent is not None:
            del denoise_mask, clean_latent
            if transformer_device.type == "cuda":
                torch.cuda.empty_cache()

        if offload_transformer_for_decode and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_offload_device)
            else:
                transformer.to(transformer_offload_device)
            logger.info("Sampling offload: moved transformer to CPU for VAE decode")
            self._cleanup_cuda(transformer_device)

        # Decode latents
        if not decode_video:
            video = None
        else:
            if offload_transformer_for_decode:
                logger.info("Sampling offload: moving VAE to GPU for decode")
                vae.to_device(transformer_device)
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)

                    # Use configured temporal tiling, or a large valid tile size if temporal splitting is disabled.
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 8192
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0

                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(
                            tile_size_in_pixels=tile_size,
                            tile_overlap_in_pixels=tile_overlap,
                        ),
                        temporal_config=TemporalTilingConfig(
                            tile_size_in_frames=effective_temporal_size,
                            tile_overlap_in_frames=effective_temporal_overlap,
                        ),
                    )
                    if temporal_tile_size > 0:
                        logger.info("Using tiled VAE decode (spatial=%dx%d, temporal=%d/%d)",
                                   tile_size, tile_overlap, temporal_tile_size, temporal_tile_overlap)
                    else:
                        logger.info("Using tiled VAE decode (spatial=%dx%d, no temporal tiling)",
                                   tile_size, tile_overlap)
                    video = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if video.dim() == 4:  # [C, T, H, W]
                        video = video.unsqueeze(0)  # [1, C, T, H, W]
                else:
                    video = vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]
                        if video.dim() == 4:  # [C, T, H, W]
                            video = video.unsqueeze(0)  # [1, C, T, H, W]

        audio_waveform = None
        loaded_audio_lazily = False
        if audio_latents is not None:
            # When no audio decoder/vocoder is loaded (subprocess mode or offloading),
            # decode audio in a separate process to avoid native crashes / OOM segfaults.
            if audio_decoder is None and vocoder is None:
                if audio_output_path and enable_audio_preview:
                    logger.info("Sampling: decoding audio via subprocess")
                    if offload_transformer_for_decode:
                        vae.to_device(original_vae_device)
                        clean_memory_on_device(transformer_device)
                    self._decode_audio_preview_subprocess(
                        audio_latents=audio_latents,
                        output_path=audio_output_path,
                        checkpoint_path=args.ltx2_checkpoint,
                    )
                    # audio_waveform stays None — the .wav was written by the subprocess
                else:
                    logger.info("Sampling: skipping audio decode (no output path or audio preview disabled)")

            elif audio_decoder is not None and vocoder is not None:
                if offload_transformer_for_decode:
                    vae.to_device(original_vae_device)
                    clean_memory_on_device(transformer_device)

                decode_device = transformer_device
                if decode_device.type == "cpu":
                    logger.info("Sampling offload: decoding audio on CPU")
                try:
                    audio_decoder.to(decode_device)
                    vocoder.to(decode_device)
                    with torch.no_grad():
                        first_param = next(audio_decoder.parameters(), None)
                        decode_dtype = first_param.dtype if first_param is not None else audio_latents.dtype
                        audio_latents = audio_latents.to(device=decode_device, dtype=decode_dtype)
                        decoded_audio = audio_decoder(audio_latents)
                        audio_waveform = vocoder(decoded_audio).squeeze(0).float().cpu()
                except Exception as exc:
                    logger.warning("Sampling: audio decode failed; skipping audio output: %s", exc)
                    audio_waveform = None
                finally:
                    audio_decoder.to("cpu")
                    vocoder.to("cpu")
            else:
                logger.warning("Sampling: audio preview requested but no decoder/vocoder available; skipping audio decode.")

        if attention_overrides:
            self._restore_attention_function(attention_overrides)
        if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_device)
            else:
                transformer.to(transformer_device)
            logger.info("Sampling offload: restored transformer to GPU after decode")
            clean_memory_on_device(transformer_device)

        # Normalize to [0, 1]
        if video is not None:
            video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        # Restore VAE state
        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video, audio_waveform

    def _do_v2v_denoising(
        self,
        latents: torch.Tensor,
        v2v_ref_latents: torch.Tensor,
        transformer,
        dit_dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor],
        sample_parameter: Dict,
        sample_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        vae,
        args: argparse.Namespace,
        offload_transformer_for_decode: bool = False,
        transformer_offload_device: Optional[torch.device] = None,
        restore_transformer_device: bool = True,
        decode_video: bool = True,
        attention_overrides=None,
    ):
        """V2V / IC-LoRA denoising: concatenate reference + target tokens with per-token timesteps.

        Mirrors the training forward pass exactly — patchify ref & target, build a ``Modality``
        with ref timesteps=0 / target timesteps=sigma, and call the base ``LTXModel`` directly
        (bypassing the LTX2Wrapper).
        """
        from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler
        from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper
        from musubi_tuner.ltx_2.model.transformer.modality import Modality
        from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape

        transformer_device = latents.device
        transformer_offload_device = transformer_offload_device or torch.device("cpu")
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)

        patchifier = VideoLatentPatchifier(patch_size=1)
        stepper = EulerDiffusionStep()

        # Prepare reference latents
        v2v_ref_latents = v2v_ref_latents.to(device=transformer_device, dtype=dit_dtype)
        bsz = latents.shape[0]
        ref_frames = int(v2v_ref_latents.shape[2])
        tgt_frames = int(latents.shape[2])
        ref_height = int(v2v_ref_latents.shape[3])
        ref_width = int(v2v_ref_latents.shape[4])
        tgt_height = int(latents.shape[3])
        tgt_width = int(latents.shape[4])

        if ref_height == tgt_height and ref_width == tgt_width:
            reference_downscale_factor = 1
        else:
            h_ratio = tgt_height / ref_height
            w_ratio = tgt_width / ref_width
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"V2V spatial mismatch: target HxW={tgt_height}x{tgt_width} vs ref HxW={ref_height}x{ref_width}. "
                    f"Ratios h={h_ratio:.2f} w={w_ratio:.2f} are not consistent integer downscale factors."
                )
            reference_downscale_factor = round(h_ratio)

        # Patchify reference tokens (constant across denoising steps)
        ref_tokens = patchifier.patchify(v2v_ref_latents)  # [B, ref_seq, D]
        ref_seq_len = ref_tokens.shape[1]

        # Conditioning mask: ref=True (conditioned, t=0), target=False (denoised, t=sigma)
        ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=transformer_device, dtype=torch.bool)

        # Compute position embeddings (constant across steps)
        ref_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=int(v2v_ref_latents.shape[1]),
                frames=ref_frames,
                height=ref_height,
                width=ref_width,
            ),
            device=transformer_device,
        )
        frame_rate_v2v = float(sample_parameter.get("frame_rate", 25))
        ref_positions = get_pixel_coords(
            latent_coords=ref_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=dit_dtype)
        ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / frame_rate_v2v
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor
            ref_positions[:, 2, ...] *= reference_downscale_factor

        tgt_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=int(latents.shape[1]),
                frames=tgt_frames,
                height=tgt_height,
                width=tgt_width,
            ),
            device=transformer_device,
        )
        tgt_positions = get_pixel_coords(
            latent_coords=tgt_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=dit_dtype)
        tgt_positions[:, 0, ...] = tgt_positions[:, 0, ...] / frame_rate_v2v

        combined_positions = torch.cat([ref_positions, tgt_positions], dim=2)

        # Get base model (bypass LTX2Wrapper)
        base_model = transformer.model if hasattr(transformer, "model") else transformer

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(base_model)
        elif getattr(args, "nf4_base", False):
            self._ensure_nf4_buffers_on_device(base_model)

        # Scheduler
        ltx2_scheduler = LTX2Scheduler()
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)

        # V2V denoising loop
        logger.info("V2V sampling: %d steps, ref_frames=%d, target_frames=%d", sample_steps, ref_frames, tgt_frames)
        with torch.no_grad():
            for step_idx in tqdm(range(len(sigmas) - 1), desc="V2V preview", leave=False):
                sigma = sigmas[step_idx]

                # Patchify current noisy target
                target_tokens = patchifier.patchify(latents.to(dtype=dit_dtype))
                target_seq_len = target_tokens.shape[1]

                # Concatenate ref + target
                combined_tokens = torch.cat([ref_tokens, target_tokens], dim=1)

                # Target conditioning mask (all False = all denoised)
                target_conditioning_mask = torch.zeros(
                    (bsz, target_seq_len), device=transformer_device, dtype=torch.bool
                )
                conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

                # Per-token timesteps: ref=0, target=sigma
                combined_timesteps = sigma.view(1, 1).expand(bsz, ref_seq_len + target_seq_len)
                combined_timesteps = torch.where(
                    conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps
                )

                perturbations = BatchedPerturbationConfig.empty(bsz)

                if do_classifier_free_guidance:
                    # Duplicate everything for CFG (unconditional + conditional)
                    cfg_tokens = combined_tokens.repeat(2, 1, 1)
                    cfg_timesteps = combined_timesteps.repeat(2, 1)
                    cfg_positions = combined_positions.repeat(2, 1, 1)
                    cfg_perturbations = BatchedPerturbationConfig.empty(bsz * 2)

                    video_modality = Modality(
                        enabled=True,
                        latent=cfg_tokens,
                        timesteps=cfg_timesteps,
                        positions=cfg_positions,
                        context=prompt_embeds,  # already [neg+pos, seq, dim] from CFG setup
                        sigma=sigma,
                        context_mask=prompt_mask,
                    )
                    pred_tokens, _ = base_model(video_modality, None, cfg_perturbations)

                    # Split and extract target predictions only
                    pred_tokens = pred_tokens[:, ref_seq_len:, :]
                    vel_uncond, vel_cond = pred_tokens.chunk(2)

                    # Unpatchify to 5D for x0 conversion
                    vel_uncond_5d = patchifier.unpatchify(
                        vel_uncond,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)
                    vel_cond_5d = patchifier.unpatchify(
                        vel_cond,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)

                    x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond_5d, sigma)
                    x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond_5d, sigma)

                    effective_cfg = float(
                        getattr(args, "video_cfg_scale", None)
                        if getattr(args, "video_cfg_scale", None) is not None
                        else (cfg_scale if cfg_scale is not None else guidance_scale)
                    )
                    video_x0 = x0_uncond + effective_cfg * (x0_cond - x0_uncond)
                    video_rescale = float(
                        getattr(args, "video_rescale_scale", None)
                        if getattr(args, "video_rescale_scale", None) is not None
                        else getattr(args, "rescale_scale", 0.0)
                    )
                    if video_rescale > 0.0:
                        pred_std = video_x0.std()
                        if pred_std > 1e-6:
                            factor = x0_cond.std() / pred_std
                            factor = video_rescale * factor + (1.0 - video_rescale)
                            video_x0 = video_x0 * factor
                else:
                    video_modality = Modality(
                        enabled=True,
                        latent=combined_tokens,
                        timesteps=combined_timesteps,
                        positions=combined_positions,
                        context=prompt_embeds,
                        sigma=sigma,
                        context_mask=prompt_mask,
                    )
                    pred_tokens, _ = base_model(video_modality, None, perturbations)

                    # Extract target predictions only
                    target_pred = pred_tokens[:, ref_seq_len:, :]
                    target_pred_5d = patchifier.unpatchify(
                        target_pred,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_height, width=tgt_width,
                        ),
                    ).to(dtype=latents.dtype)

                    video_x0 = X0PredictionWrapper.velocity_to_x0(latents, target_pred_5d, sigma)

                # Euler step
                latents = stepper.step(latents, video_x0, sigmas, step_idx)

        # Offload transformer for VAE decode
        if offload_transformer_for_decode and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_offload_device)
            else:
                transformer.to(transformer_offload_device)
            logger.info("V2V sampling offload: moved transformer to CPU for VAE decode")
            self._cleanup_cuda(transformer_device)

        # Decode latents
        if not decode_video:
            video = None
        else:
            if offload_transformer_for_decode:
                vae.to_device(transformer_device)
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 8192
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0
                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=tile_overlap),
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=effective_temporal_size, tile_overlap_in_frames=effective_temporal_overlap),
                    )
                    video = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                else:
                    video = vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]
                        if video.dim() == 4:
                            video = video.unsqueeze(0)

        if attention_overrides:
            self._restore_attention_function(attention_overrides)
        if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_device)
            else:
                transformer.to(transformer_device)
            logger.info("V2V sampling offload: restored transformer to GPU after decode")
            self._cleanup_cuda(transformer_device)

        if video is not None:
            video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video, None  # no audio for v2v sampling

    def _do_av_ic_denoising(
        self,
        latents: torch.Tensor,
        v2v_ref_latents: torch.Tensor,
        ref_audio_latents: Optional[torch.Tensor],
        transformer,
        dit_dtype: torch.dtype,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor],
        sample_parameter: Dict,
        sample_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: Optional[float],
        vae,
        audio_decoder,
        vocoder,
        args: argparse.Namespace,
        offload_transformer_for_decode: bool = False,
        transformer_offload_device: Optional[torch.device] = None,
        restore_transformer_device: bool = True,
        decode_video: bool = True,
        attention_overrides=None,
    ):
        """AV IC denoising for `av_ic` and `video_ref_only_av`.

        Video always uses reference+target tokens. Audio uses reference+target tokens
        for `av_ic`, or target-only audio for `video_ref_only_av`.
        """
        from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler
        from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper
        from musubi_tuner.ltx_2.model.transformer.modality import Modality
        from musubi_tuner.ltx_2.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
        from musubi_tuner.networks.lora_ltx2 import _split_av_context

        transformer_device = latents.device
        transformer_offload_device = transformer_offload_device or torch.device("cpu")
        original_vae_device = getattr(vae, "device", torch.device("cpu"))
        original_vae_dtype = getattr(vae, "dtype", torch.float32)
        resolved_ic_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
        av_cross_attention_mode = _normalize_av_cross_attention_mode(
            getattr(args, "av_cross_attention_mode", "both")
        )
        av_ic_a2v_enabled = av_cross_attention_mode in {"both", "a2v_only"}
        av_ic_v2a_enabled = av_cross_attention_mode in {"both", "v2a_only"}

        video_patchifier = VideoLatentPatchifier(patch_size=1)
        stepper = EulerDiffusionStep()
        bsz = latents.shape[0]
        frame_rate_sample = float(sample_parameter.get("frame_rate", 25))

        base_model = transformer.model if hasattr(transformer, "model") else transformer

        # --- Video reference preparation (constant across steps) ---
        ref_video_list = self._normalize_reference_tensor_collection(v2v_ref_latents, expected_ndim=5)
        if not ref_video_list:
            raise ValueError("AV IC sampling requires at least one reference video latent tensor")
        v2v_ref_latents = torch.cat(
            [ref.to(device=transformer_device, dtype=dit_dtype) for ref in ref_video_list],
            dim=2,
        )
        ref_video_tokens = video_patchifier.patchify(v2v_ref_latents)
        ref_video_seq_len = ref_video_tokens.shape[1]
        ref_video_cond_mask = torch.ones((bsz, ref_video_seq_len), device=transformer_device, dtype=torch.bool)

        ref_frames = int(v2v_ref_latents.shape[2])
        tgt_frames = int(latents.shape[2])
        ref_h, ref_w = int(v2v_ref_latents.shape[3]), int(v2v_ref_latents.shape[4])
        tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
        if ref_h == tgt_h and ref_w == tgt_w:
            reference_downscale_factor = 1
        else:
            h_ratio = tgt_h / ref_h
            w_ratio = tgt_w / ref_w
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"av_ic sampling requires integer ref-video downscale. "
                    f"Got ref={ref_h}x{ref_w}, target={tgt_h}x{tgt_w}, "
                    f"h_ratio={h_ratio:.2f}, w_ratio={w_ratio:.2f}"
                )
            reference_downscale_factor = round(h_ratio)

        ref_video_coords = video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz, channels=int(v2v_ref_latents.shape[1]),
                frames=ref_frames, height=ref_h, width=ref_w,
            ),
            device=transformer_device,
        )
        ref_video_pos = get_pixel_coords(
            latent_coords=ref_video_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
        ).to(dtype=dit_dtype)
        ref_video_pos[:, 0, ...] = ref_video_pos[:, 0, ...] / frame_rate_sample
        if reference_downscale_factor != 1:
            ref_video_pos = ref_video_pos.clone()
            ref_video_pos[:, 1, ...] *= reference_downscale_factor
            ref_video_pos[:, 2, ...] *= reference_downscale_factor

        tgt_video_coords = video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz, channels=int(latents.shape[1]),
                frames=tgt_frames, height=tgt_h, width=tgt_w,
            ),
            device=transformer_device,
        )
        tgt_video_pos = get_pixel_coords(
            latent_coords=tgt_video_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
        ).to(dtype=dit_dtype)
        tgt_video_pos[:, 0, ...] = tgt_video_pos[:, 0, ...] / frame_rate_sample
        video_combined_pos = torch.cat([ref_video_pos, tgt_video_pos], dim=2)

        audio_patchifier = getattr(transformer, "_audio_patchifier", None)
        if audio_patchifier is None and hasattr(transformer, "module"):
            audio_patchifier = getattr(transformer.module, "_audio_patchifier", None)
        if audio_patchifier is None and hasattr(transformer, "model"):
            audio_patchifier = getattr(transformer.model, "_audio_patchifier", None)
        if audio_patchifier is None:
            raise ValueError("AV IC sampling requires an audio patchifier on the model")

        ref_audio_tokens = None
        ref_audio_pos = None
        ref_audio_seq_len = 0

        ref_audio_list = self._normalize_reference_tensor_collection(ref_audio_latents, expected_ndim=4)
        if ref_audio_list:
            ref_audio_latents = torch.cat(
                [ref.to(device=transformer_device, dtype=dit_dtype) for ref in ref_audio_list],
                dim=2,
            )
            ref_audio_tokens = audio_patchifier.patchify(ref_audio_latents)
            ref_audio_seq_len = ref_audio_tokens.shape[1]
            channels_audio = int(ref_audio_latents.shape[1])
            mel_bins = int(ref_audio_latents.shape[3])

            use_negative_positions = bool(getattr(args, "audio_ref_use_negative_positions", False))
            ref_audio_shape = AudioLatentShape(
                batch=bsz,
                channels=channels_audio,
                frames=ref_audio_seq_len,
                mel_bins=mel_bins,
            )
            ref_audio_pos = audio_patchifier.get_patch_grid_bounds(ref_audio_shape, device=transformer_device).to(dtype=dit_dtype)
            if use_negative_positions:
                _hop = getattr(audio_patchifier, "hop_length", 160)
                _ds = getattr(audio_patchifier, "audio_latent_downsample_factor", 4)
                _sr = getattr(audio_patchifier, "sample_rate", 16000)
                time_per_latent = float(_hop) * float(_ds) / float(_sr)
                ref_duration = ref_audio_pos[:, :, -1:, 1:2]
                ref_audio_pos = ref_audio_pos - ref_duration - time_per_latent

            audio_length = int(sample_parameter.get("audio_latent_frames", ref_audio_latents.shape[2]))
        else:
            audio_cfg = self._get_audio_preview_config(args, transformer)
            channels_audio = int(audio_cfg["channels"])
            mel_bins = int(audio_cfg["mel_bins"])
            sample_width = int(sample_parameter.get("width", 768))
            sample_height = int(sample_parameter.get("height", 512))
            sample_frame_count = int(sample_parameter.get("frame_count", max(tgt_frames, 1)))
            sample_rate = int(audio_cfg["sample_rate"])
            hop_length = int(audio_cfg["hop_length"])
            audio_downsample = int(audio_cfg["audio_latent_downsample_factor"])
            video_shape = VideoPixelShape(
                batch=bsz,
                frames=sample_frame_count,
                height=sample_height,
                width=sample_width,
                fps=frame_rate_sample,
            )
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                video_shape,
                channels=channels_audio,
                mel_bins=mel_bins,
                sample_rate=sample_rate,
                hop_length=hop_length,
                audio_latent_downsample_factor=audio_downsample,
            )
            audio_length = max(int(sample_parameter.get("audio_latent_frames", int(audio_shape.frames))), 1)

        audio_latents = torch.randn(
            (bsz, channels_audio, audio_length, mel_bins),
            dtype=torch.float32,
            device=transformer_device,
        )

        # Target audio positions
        tgt_audio_seq_len = audio_length
        tgt_audio_shape = AudioLatentShape(batch=bsz, channels=channels_audio, frames=tgt_audio_seq_len, mel_bins=mel_bins)
        tgt_audio_pos = audio_patchifier.get_patch_grid_bounds(tgt_audio_shape, device=transformer_device).to(dtype=dit_dtype)
        audio_combined_pos = (
            torch.cat([ref_audio_pos, tgt_audio_pos], dim=2)
            if ref_audio_pos is not None
            else tgt_audio_pos
        )

        # Context splitting (done per-step if CFG is enabled due to batch doubling)
        if not do_classifier_free_guidance:
            video_ctx, audio_ctx = _split_av_context(base_model, prompt_embeds)

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(base_model)
        elif getattr(args, "nf4_base", False):
            self._ensure_nf4_buffers_on_device(base_model)

        # Scheduler
        ltx2_scheduler = LTX2Scheduler()
        sigmas = ltx2_scheduler.execute(steps=sample_steps).to(device=transformer_device, dtype=torch.float32)

        logger.info(
            "AV_IC sampling: %d steps, ref_video_frames=%d, tgt_video_frames=%d, ref_audio_T=%d, tgt_audio_T=%d",
            sample_steps, ref_frames, tgt_frames, ref_audio_seq_len, tgt_audio_seq_len,
        )

        with torch.no_grad():
            for step_idx in tqdm(range(len(sigmas) - 1), desc="AV_IC preview", leave=False):
                sigma = sigmas[step_idx]

                # --- Video: patchify current target, concatenate with ref ---
                tgt_video_tokens = video_patchifier.patchify(latents.to(dtype=dit_dtype))
                tgt_video_seq = tgt_video_tokens.shape[1]
                video_tokens = torch.cat([ref_video_tokens, tgt_video_tokens], dim=1)

                tgt_video_cond = torch.zeros((bsz, tgt_video_seq), device=transformer_device, dtype=torch.bool)
                video_cond = torch.cat([ref_video_cond_mask, tgt_video_cond], dim=1)
                video_ts = sigma.view(1, 1).expand(bsz, ref_video_seq_len + tgt_video_seq)
                video_ts = torch.where(video_cond, torch.zeros_like(video_ts), video_ts)

                # --- Audio: patchify current target, concatenate with ref ---
                tgt_audio_tokens = audio_patchifier.patchify(audio_latents.to(dtype=dit_dtype))
                tgt_audio_ts = sigma.view(1, 1).expand(bsz, tgt_audio_seq_len).to(dtype=dit_dtype)
                if ref_audio_tokens is not None:
                    audio_tokens = torch.cat([ref_audio_tokens, tgt_audio_tokens], dim=1)
                    ref_audio_ts = torch.zeros((bsz, ref_audio_seq_len), device=transformer_device, dtype=dit_dtype)
                    audio_ts = torch.cat([ref_audio_ts, tgt_audio_ts], dim=1)
                else:
                    audio_tokens = tgt_audio_tokens
                    audio_ts = tgt_audio_ts

                total_video_seq = ref_video_seq_len + tgt_video_seq
                total_audio_seq = audio_tokens.shape[1]
                mask_dtype = dit_dtype if dit_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32
                neg_inf = torch.finfo(mask_dtype).min
                a2v_mask = None
                if not av_ic_a2v_enabled:
                    a2v_mask = torch.full((bsz, total_video_seq, total_audio_seq), neg_inf, device=transformer_device, dtype=mask_dtype)
                v2a_mask = None
                if not av_ic_v2a_enabled:
                    v2a_mask = torch.full((bsz, total_audio_seq, total_video_seq), neg_inf, device=transformer_device, dtype=mask_dtype)

                if do_classifier_free_guidance:
                    # Duplicate for CFG: [unconditional, conditional]
                    cfg_video_tokens = video_tokens.repeat(2, 1, 1)
                    cfg_video_ts = video_ts.repeat(2, 1)
                    cfg_video_pos = video_combined_pos.repeat(2, 1, 1, 1)
                    cfg_audio_tokens = audio_tokens.repeat(2, 1, 1)
                    cfg_audio_ts = audio_ts.repeat(2, 1)
                    cfg_audio_pos = audio_combined_pos.repeat(2, 1, 1, 1)
                    cfg_a2v_mask = a2v_mask.repeat(2, 1, 1) if a2v_mask is not None else None
                    cfg_v2a_mask = v2a_mask.repeat(2, 1, 1) if v2a_mask is not None else None
                    cfg_perturbations = BatchedPerturbationConfig.empty(bsz * 2)

                    # prompt_embeds is already [neg+pos, seq, dim] from CFG setup
                    cfg_video_ctx, cfg_audio_ctx = _split_av_context(base_model, prompt_embeds)

                    video_modality = Modality(
                        enabled=True,
                        latent=cfg_video_tokens,
                        timesteps=cfg_video_ts,
                        positions=cfg_video_pos,
                        context=cfg_video_ctx,
                        sigma=sigma,
                        context_mask=prompt_mask,
                        a2v_cross_attention_mask=cfg_a2v_mask,
                    )
                    audio_modality = Modality(
                        enabled=True,
                        latent=cfg_audio_tokens,
                        timesteps=cfg_audio_ts,
                        positions=cfg_audio_pos,
                        context=cfg_audio_ctx,
                        sigma=sigma,
                        v2a_cross_attention_mask=cfg_v2a_mask,
                    )

                    video_pred_all, audio_pred_all = base_model(video_modality, audio_modality, cfg_perturbations)

                    # Extract target predictions and split unconditional/conditional
                    video_vel_all = video_pred_all[:, ref_video_seq_len:, :]
                    audio_vel_all = audio_pred_all[:, ref_audio_seq_len:, :]
                    video_vel_uncond, video_vel_cond = video_vel_all.chunk(2)
                    audio_vel_uncond, audio_vel_cond = audio_vel_all.chunk(2)

                    # Unpatchify for x0 conversion
                    def _unpatchify_video(vel):
                        return video_patchifier.unpatchify(
                            vel,
                            output_shape=VideoLatentShape(
                                batch=bsz, channels=int(latents.shape[1]),
                                frames=tgt_frames, height=tgt_h, width=tgt_w,
                            ),
                        ).to(dtype=latents.dtype)

                    def _unpatchify_audio(vel):
                        return audio_patchifier.unpatchify(
                            vel,
                            output_shape=AudioLatentShape(
                                batch=bsz, channels=channels_audio,
                                frames=tgt_audio_seq_len, mel_bins=mel_bins,
                            ),
                        ).to(dtype=audio_latents.dtype)

                    effective_video_cfg = float(
                        getattr(args, "video_cfg_scale", None)
                        if getattr(args, "video_cfg_scale", None) is not None
                        else (cfg_scale if cfg_scale is not None else guidance_scale)
                    )
                    effective_audio_cfg = float(
                        getattr(args, "audio_cfg_scale", None)
                        if getattr(args, "audio_cfg_scale", None) is not None
                        else (cfg_scale if cfg_scale is not None else guidance_scale)
                    )

                    v_x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, _unpatchify_video(video_vel_uncond), sigma)
                    v_x0_cond = X0PredictionWrapper.velocity_to_x0(latents, _unpatchify_video(video_vel_cond), sigma)
                    video_x0 = v_x0_uncond + effective_video_cfg * (v_x0_cond - v_x0_uncond)

                    a_x0_uncond = X0PredictionWrapper.velocity_to_x0(audio_latents, _unpatchify_audio(audio_vel_uncond), sigma)
                    a_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, _unpatchify_audio(audio_vel_cond), sigma)
                    audio_x0 = a_x0_uncond + effective_audio_cfg * (a_x0_cond - a_x0_uncond)

                    video_rescale = float(
                        getattr(args, "video_rescale_scale", None)
                        if getattr(args, "video_rescale_scale", None) is not None
                        else getattr(args, "rescale_scale", 0.0)
                    )
                    audio_rescale = float(
                        getattr(args, "audio_rescale_scale", None)
                        if getattr(args, "audio_rescale_scale", None) is not None
                        else getattr(args, "rescale_scale", 0.0)
                    )
                    if video_rescale > 0.0:
                        pred_std = video_x0.std()
                        if pred_std > 1e-6:
                            factor = v_x0_cond.std() / pred_std
                            factor = video_rescale * factor + (1.0 - video_rescale)
                            video_x0 = video_x0 * factor
                    if audio_rescale > 0.0:
                        pred_std = audio_x0.std()
                        if pred_std > 1e-6:
                            factor = a_x0_cond.std() / pred_std
                            factor = audio_rescale * factor + (1.0 - audio_rescale)
                            audio_x0 = audio_x0 * factor
                else:
                    perturbations = BatchedPerturbationConfig.empty(bsz)

                    video_modality = Modality(
                        enabled=True,
                        latent=video_tokens,
                        timesteps=video_ts,
                        positions=video_combined_pos,
                        context=video_ctx,
                        sigma=sigma,
                        context_mask=prompt_mask,
                        a2v_cross_attention_mask=a2v_mask,
                    )
                    audio_modality = Modality(
                        enabled=True,
                        latent=audio_tokens,
                        timesteps=audio_ts,
                        positions=audio_combined_pos,
                        context=audio_ctx,
                        sigma=sigma,
                        v2a_cross_attention_mask=v2a_mask,
                    )

                    video_pred_all, audio_pred_all = base_model(video_modality, audio_modality, perturbations)

                    # Extract target-only predictions
                    video_vel = video_pred_all[:, ref_video_seq_len:, :]
                    audio_vel = audio_pred_all[:, ref_audio_seq_len:, :]

                    video_vel_5d = video_patchifier.unpatchify(
                        video_vel,
                        output_shape=VideoLatentShape(
                            batch=bsz, channels=int(latents.shape[1]),
                            frames=tgt_frames, height=tgt_h, width=tgt_w,
                        ),
                    ).to(dtype=latents.dtype)
                    audio_vel_4d = audio_patchifier.unpatchify(
                        audio_vel,
                        output_shape=AudioLatentShape(
                            batch=bsz, channels=channels_audio,
                            frames=tgt_audio_seq_len, mel_bins=mel_bins,
                        ),
                    ).to(dtype=audio_latents.dtype)

                    video_x0 = X0PredictionWrapper.velocity_to_x0(latents, video_vel_5d, sigma)
                    audio_x0 = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_vel_4d, sigma)

                # Euler step for both
                latents = stepper.step(latents, video_x0, sigmas, step_idx)
                audio_latents = stepper.step(audio_latents, audio_x0, sigmas, step_idx)

        # --- Decode video ---
        if offload_transformer_for_decode and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_offload_device)
            else:
                transformer.to(transformer_offload_device)
            logger.info("AV_IC sampling offload: moved transformer to CPU for decode")
            self._cleanup_cuda(transformer_device)

        video = None
        if decode_video:
            if offload_transformer_for_decode:
                vae.to_device(transformer_device)
            with torch.no_grad():
                use_tiled_vae = getattr(args, "sample_tiled_vae", False)
                if use_tiled_vae:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
                    tile_size = getattr(args, "sample_vae_tile_size", 512)
                    tile_overlap = getattr(args, "sample_vae_tile_overlap", 64)
                    temporal_tile_size = getattr(args, "sample_vae_temporal_tile_size", 0)
                    temporal_tile_overlap = getattr(args, "sample_vae_temporal_tile_overlap", 8)
                    effective_temporal_size = temporal_tile_size if temporal_tile_size > 0 else 8192
                    effective_temporal_overlap = temporal_tile_overlap if temporal_tile_size > 0 else 0
                    tiling_config = TilingConfig(
                        spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=tile_overlap),
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=effective_temporal_size, tile_overlap_in_frames=effective_temporal_overlap),
                    )
                    video = vae.tiled_decode(latents.squeeze(0), tiling_config)
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                else:
                    video = vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]
                        if video.dim() == 4:
                            video = video.unsqueeze(0)

        # --- Decode audio ---
        audio_waveform = None
        if audio_decoder is not None and vocoder is not None:
            if offload_transformer_for_decode:
                vae.to_device(original_vae_device)
                clean_memory_on_device(transformer_device)
            decode_device = transformer_device
            try:
                audio_decoder.to(decode_device)
                vocoder.to(decode_device)
                with torch.no_grad():
                    first_param = next(audio_decoder.parameters(), None)
                    decode_dtype = first_param.dtype if first_param is not None else audio_latents.dtype
                    decoded_audio = audio_decoder(audio_latents.to(device=decode_device, dtype=decode_dtype))
                    audio_waveform = vocoder(decoded_audio).squeeze(0).float().cpu()
            except Exception as exc:
                logger.warning("AV_IC sampling: audio decode failed: %s", exc)
                audio_waveform = None
            finally:
                audio_decoder.to("cpu")
                vocoder.to("cpu")
        elif audio_latents is not None:
            # Subprocess decode fallback
            audio_output_path = sample_parameter.get("audio_output_path")
            if audio_output_path:
                try:
                    self._decode_audio_preview_subprocess(
                        audio_latents=audio_latents,
                        output_path=audio_output_path,
                        checkpoint_path=args.ltx2_checkpoint,
                    )
                except Exception as exc:
                    logger.warning("AV_IC sampling: subprocess audio decode failed: %s", exc)

        if attention_overrides:
            self._restore_attention_function(attention_overrides)
        if offload_transformer_for_decode and restore_transformer_device and transformer_device != transformer_offload_device:
            if hasattr(transformer, "move_to_device_except_swap_blocks"):
                transformer.move_to_device_except_swap_blocks(transformer_device)
            else:
                transformer.to(transformer_device)
            logger.info("AV_IC sampling offload: restored transformer to GPU after decode")
            self._cleanup_cuda(transformer_device)

        if video is not None:
            video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).to("cpu")

        vae.to_device(original_vae_device)
        vae.to_dtype(original_vae_dtype)

        return video, audio_waveform

    def do_inference_two_stage(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: Dict,
        vae,
        dit_dtype: torch.dtype,
        transformer,
        width: int,
        height: int,
        frame_count: int,
        sample_steps: int,
        guidance_scale: float,
        cfg_scale: Optional[float],
        seed: Optional[int],
        generator: torch.Generator,
        spatial_upsampler_path: str,
        distilled_lora_path: Optional[str] = None,
        stage2_steps: int = 4,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
        enable_audio_preview: bool = False,
        decode_video: bool = True,
        audio_only: bool = False,
        conditioning_latent: Optional[torch.Tensor] = None,
    ):
        """Generate sample video using two-stage inference (half-res + upsample + refine)."""
        device = accelerator.device

        # Create inferencer
        inferencer = LTX2Inferencer(
            transformer=transformer,
            vae=vae,
            device=device,
            dit_dtype=dit_dtype,
            audio_video_mode=self._audio_video,
        )

        # Load upsampler
        inferencer.load_spatial_upsampler(spatial_upsampler_path, device=torch.device("cpu"))

        # Load distilled LoRA if provided
        if distilled_lora_path:
            inferencer.load_distilled_lora(distilled_lora_path)

        # Get prompt embeddings from sample_parameter
        prompt_embeds = sample_parameter.get("prompt_embeds")
        prompt_mask = sample_parameter.get("prompt_attention_mask")
        negative_embeds = sample_parameter.get("negative_prompt_embeds")
        negative_mask = sample_parameter.get("negative_prompt_attention_mask")

        # Build audio config if needed
        audio_config = None
        if enable_audio_preview and self._audio_video:
            audio_config = self._get_audio_preview_config(args, transformer)

        # Prepare tiled VAE config
        tiled_vae_config = None
        if getattr(args, "sample_tiled_vae", False):
            tiled_vae_config = {
                "tile_size": getattr(args, "sample_vae_tile_size", 512),
                "tile_overlap": getattr(args, "sample_vae_tile_overlap", 64),
                "temporal_tile_size": getattr(args, "sample_vae_temporal_tile_size", 0) or 8192,
                "temporal_tile_overlap": getattr(args, "sample_vae_temporal_tile_overlap", 8),
            }

        # Build inference config
        config = InferenceConfig(
            prompt=sample_parameter.get("prompt", ""),
            negative_prompt=sample_parameter.get("negative_prompt"),
            width=width,
            height=height,
            frame_count=frame_count,
            frame_rate=sample_parameter.get("frame_rate", 25.0),
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            cfg_scale=cfg_scale,
            video_cfg_scale=getattr(args, "video_cfg_scale", None),
            audio_cfg_scale=getattr(args, "audio_cfg_scale", None),
            seed=seed,
            two_stage=True,
            spatial_upsampler_path=spatial_upsampler_path,
            distilled_lora_path=distilled_lora_path,
            stage2_steps=stage2_steps,
            enable_audio=enable_audio_preview,
            audio_only=audio_only,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_attention_mask=negative_mask,
            conditioning_latent=conditioning_latent,
            use_i2v_token_timestep_mask=bool(getattr(args, "sample_i2v_token_timestep_mask", True)),
            offload_between_stages=bool(getattr(args, "sample_with_offloading", False)),
            stg_scale=float(getattr(args, "stg_scale", 0.0) or 0.0),
            stg_blocks=getattr(args, "stg_blocks", None),
            stg_mode=str(getattr(args, "stg_mode", "video")),
            rescale_scale=float(getattr(args, "rescale_scale", 0.0) or 0.0),
            video_rescale_scale=getattr(args, "video_rescale_scale", None),
            audio_rescale_scale=getattr(args, "audio_rescale_scale", None),
            video_modality_scale=float(getattr(args, "video_modality_scale", 1.0) or 1.0),
            audio_modality_scale=float(getattr(args, "audio_modality_scale", 1.0) or 1.0),
            extra={"audio_config": audio_config} if audio_config else {},
        )

        # Disable flash attention for sampling if requested
        attention_overrides = []
        if getattr(args, "sample_disable_flash_attn", True):
            from musubi_tuner.ltx_2.model.transformer.attention import AttentionFunction
            logger.info("Two-stage sampling: disabling FlashAttention for preview")
            attention_overrides = self._override_attention_function(
                transformer, AttentionFunction.PYTORCH
            )

        try:
            # Run two-stage inference
            video, audio_waveform = inferencer.generate(
                config=config,
                audio_decoder=audio_decoder,
                vocoder=vocoder,
                decode_video=decode_video,
                use_tiled_vae=bool(tiled_vae_config),
                tiled_vae_config=tiled_vae_config,
            )
        finally:
            # Restore attention settings
            if attention_overrides:
                self._restore_attention_function(attention_overrides)

        return video, audio_waveform
