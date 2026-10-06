from __future__ import annotations

import json
import hashlib
import logging
import math
import random
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from musubi_tuner.minimax_h3.architecture import temporal_shape
from musubi_tuner.minimax_h3.cache import H3_TEXT_HIDDEN_KEY, H3_TEXT_TOKEN_TAGS_KEY
from musubi_tuner.minimax_h3.component_loader import load_audio_vae_decoder, load_video_vae_decoder
from musubi_tuner.minimax_h3.inference import (
    AUDIO_FLOW_SHIFT,
    VIDEO_FLOW_SHIFT,
    decode_latents_sequentially,
    denoise_fl2va,
    save_av_mp4,
    shifted_flow_schedule,
)
from musubi_tuner.minimax_h3.integration import _NativeGenerator
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.minimax_h3.packing import (
    build_row_timesteps,
    build_t2va_packed_sequence,
    pack_audio_latents,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from musubi_tuner.minimax_h3.va_judger import (
    VAJudgerClient,
    dimension_normalized_advantages,
    nft_mixture_loss,
    reward_components,
)
from musubi_tuner.networks import lora_minimax_h3

logger = logging.getLogger(__name__)

_RESUME_RUNTIME_KEYS = {
    "device",
    "initial_lora",
    "max_steps",
    "output_dir",
    "resume",
    "reward_pair_batch_size",
    "reward_endpoint",
    "reward_timeout",
    "save_every",
    "blocks_to_swap",
    "block_swap_h2d_only",
    "use_pinned_memory_for_block_swap",
}


@dataclass(frozen=True)
class VAJudgerTrainConfig:
    model: Path
    text_encoder: Path
    tokenizer: Path
    video_vae: Path
    audio_vae: Path
    prompts: Path
    output_dir: Path
    reward_endpoint: str | None = None
    max_steps: int = 100
    group_size: int = 4
    duration: int = 5
    height: int = 512
    width: int = 896
    inference_steps: int = 20
    learning_rate: float = 3e-5
    network_dim: int = 32
    network_alpha: float = 32.0
    beta_mix: float = 1.0
    kl_beta: float = 1e-4
    advantage_clip: float = 5.0
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    save_every: int = 10
    seed: int = 42
    device: str | None = None
    dtype: str = "bfloat16"
    initial_lora: Path | None = None
    resume: Path | None = None
    fp8_base: bool = False
    int8_convrot_base: bool = False
    blocks_to_swap: int = 0
    block_swap_h2d_only: bool = False
    use_pinned_memory_for_block_swap: bool = False
    token_refiner: bool = True
    text_encoder_quantization: str = "int8"
    reward_pair_batch_size: int = 1
    reward_timeout: float = 900.0

    def validate(self) -> None:
        required = (self.model, self.text_encoder, self.tokenizer, self.video_vae, self.audio_vae, self.prompts)
        missing = [str(path) for path in required if not Path(path).exists()]
        missing.extend(str(path) for path in (self.initial_lora, self.resume) if path is not None and not Path(path).is_file())
        if missing:
            raise ValueError("missing required input(s): " + ", ".join(missing))
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.max_steps < 1 or self.inference_steps < 2:
            raise ValueError("max_steps must be positive and inference_steps must be at least 2")
        if self.duration < 1 or self.height < 64 or self.width < 64:
            raise ValueError("duration and canvas dimensions must be positive")
        if self.height % 64 or self.width % 64:
            raise ValueError("height and width must be multiples of 64")
        if self.learning_rate <= 0 or self.network_dim < 1 or self.network_alpha <= 0:
            raise ValueError("learning_rate, network_dim, and network_alpha must be positive")
        if self.beta_mix <= 0 or self.kl_beta < 0 or self.advantage_clip <= 0:
            raise ValueError("invalid NFT coefficients")
        if self.video_loss_weight < 0 or self.audio_loss_weight < 0 or not (self.video_loss_weight + self.audio_loss_weight):
            raise ValueError("at least one modality loss weight must be positive")
        if self.dtype != "bfloat16":
            raise ValueError("the native H3 online runtime currently requires dtype=bfloat16")
        if self.save_every < 1:
            raise ValueError("save_every must be positive")
        if self.reward_pair_batch_size < 1:
            raise ValueError("reward_pair_batch_size must be positive")
        if not math.isfinite(self.reward_timeout) or self.reward_timeout <= 0:
            raise ValueError("reward_timeout must be positive and finite")
        numeric = (
            self.learning_rate,
            self.network_alpha,
            self.beta_mix,
            self.kl_beta,
            self.advantage_clip,
            self.video_loss_weight,
            self.audio_loss_weight,
            self.max_grad_norm,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("numeric training options must be finite")
        if self.initial_lora is not None and self.resume is not None:
            raise ValueError("initial_lora and resume are mutually exclusive")
        if not 0 <= self.blocks_to_swap <= 48:
            raise ValueError("blocks_to_swap must be in [0, 48]")
        if self.block_swap_h2d_only and not self.blocks_to_swap:
            raise ValueError("block_swap_h2d_only requires blocks_to_swap")
        if self.use_pinned_memory_for_block_swap and not self.blocks_to_swap:
            raise ValueError("use_pinned_memory_for_block_swap requires blocks_to_swap")
        if self.text_encoder_quantization not in {"none", "int8", "nf4", "nvfp4", "nvfp4_awq"}:
            raise ValueError("unsupported text_encoder_quantization")


@dataclass
class RolloutCandidate:
    prompt: str
    seed: int
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    media_path: Path


class _BackwardNativeGenerator(_NativeGenerator):
    """Native loader variant that creates one backward-capable swap offloader."""

    def _load_transformer(self):
        if not self.blocks_to_swap:
            return super()._load_transformer()

        from musubi_tuner.minimax_h3.model_loader import load_transformer

        transformer = load_transformer(
            self.model,
            mode=self.mode,
            loading_device=torch.device("cpu"),
            fp8_scaled=self.fp8_scaled,
            quantization_device=self.device if self.fp8_scaled else None,
            int8_convrot=self.int8_convrot,
            target_device=self.device,
            blocks_to_swap=self.blocks_to_swap,
            block_swap_h2d_only=self.block_swap_h2d_only,
        )
        transformer.requires_grad_(False).eval()
        if self.fused_qk_norm_rope:
            transformer.enable_fused_qk_norm_rope()
        transformer.enable_block_swap(
            self.blocks_to_swap,
            BlockSwapConfig(
                device=self.device,
                supports_backward=True,
                use_pinned_memory=self.use_pinned_memory_for_block_swap,
                h2d_only=self.block_swap_h2d_only,
                ring_size=self.block_swap_ring_size,
                granularity=self.block_swap_granularity,
            ),
        )
        transformer.move_to_device_except_swap_blocks(self.device)
        transformer.switch_block_swap_for_inference()
        return transformer, []


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if path.suffix.lower() == ".jsonl":
            try:
                value = json.loads(line)
                prompt = value["prompt"] if isinstance(value, dict) else value
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: expected JSON string or object with prompt") from exc
        else:
            prompt = line
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
        prompts.append(prompt.strip())
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    export = getattr(module, "export_state_dict", None)
    state = export() if callable(export) else module.state_dict()
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _semantic_config(config: VAJudgerTrainConfig) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in asdict(config).items():
        if key in _RESUME_RUNTIME_KEYS:
            continue
        values[key] = str(value.resolve()) if isinstance(value, Path) else value
    return values


def _config_fingerprint(values: dict[str, object]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash a policy without depending on torch's serialization details."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<Q", tensor.ndim))
        for size in tensor.shape:
            digest.update(struct.pack("<q", size))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class VAJudgerRuntime:
    """Single-GPU prompt-only online NFT loop for H3 joint video/audio."""

    def __init__(self, config: VAJudgerTrainConfig) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError("H3 VA-Judger training requires a CUDA device")
        if config.resume is None and config.output_dir.exists():
            owned = list(config.output_dir.glob("*.safetensors")) + list(config.output_dir.glob("*.resume.pt"))
            owned += list(config.output_dir.glob("*.reward.json"))
            owned += list(config.output_dir.glob("metrics.jsonl")) + list((config.output_dir / "rollouts").glob("*"))
            if owned:
                raise FileExistsError(f"output_dir contains an existing VA-Judger run: {config.output_dir}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_dir = self.config.output_dir / "rollouts"
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = load_prompts(config.prompts)
        self.prompt_fingerprint = hashlib.sha256("\n".join(self.prompts).encode("utf-8")).hexdigest()
        self.semantic_config = _semantic_config(config)
        self.semantic_fingerprint = _config_fingerprint(self.semantic_config)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.reward = None
        if config.reward_endpoint is not None:
            self.reward = VAJudgerClient(
                config.reward_endpoint,
                timeout=config.reward_timeout,
                pair_batch_size=config.reward_pair_batch_size,
            )
        self.generator = self._make_generator()
        # Qwen and the DiT do not coexist on the GPU: cache every unique prompt
        # on CPU first, then load the transformer for the training loop.
        self.conditioning = {
            prompt: {key: value.detach().cpu() for key, value in self.generator._encode_prompt(prompt).items()}
            for prompt in dict.fromkeys(self.prompts)
        }
        self.transformer, _ = self.generator._load_transformer()
        self.transformer.requires_grad_(False).eval()
        enable_checkpointing = getattr(self.transformer, "enable_gradient_checkpointing", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
        self.network = lora_minimax_h3.create_arch_network(
            1.0,
            config.network_dim,
            config.network_alpha,
            None,
            [],
            self.transformer,
            h3_lora_token_refiner=str(config.token_refiner).lower(),
        )
        self.network.apply_to(None, self.transformer, apply_text_encoder=False, apply_unet=True)
        if config.initial_lora is not None:
            incompatible = self.network.load_weights(str(config.initial_lora))
            if incompatible.missing_keys or incompatible.unexpected_keys:
                details = []
                if incompatible.missing_keys:
                    details.append(f"missing={incompatible.missing_keys[:8]}")
                if incompatible.unexpected_keys:
                    details.append(f"unexpected={incompatible.unexpected_keys[:8]}")
                raise ValueError(f"initial_lora does not match the configured H3 LoRA: {'; '.join(details)}")
        self.network.to(self.device)
        groups, _ = self.network.prepare_optimizer_params(unet_lr=config.learning_rate)
        self.optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
        self.old_policy = _cpu_state_dict(self.network)
        self.step = 0
        self.prompt_cursor = 0
        if config.resume is not None:
            self.load_resume(config.resume)

    def _make_generator(self) -> _NativeGenerator:
        return _BackwardNativeGenerator(
            model=self.config.model,
            text_encoder=self.config.text_encoder,
            tokenizer=self.config.tokenizer,
            video_vae=self.config.video_vae,
            audio_vae=self.config.audio_vae,
            device=self.device,
            num_inference_steps=self.config.inference_steps,
            height=self.config.height,
            width=self.config.width,
            fp8_scaled=self.config.fp8_base,
            int8_convrot=self.config.int8_convrot_base,
            text_encoder_quantization=self.config.text_encoder_quantization,
            blocks_to_swap=self.config.blocks_to_swap,
            block_swap_h2d_only=self.config.block_swap_h2d_only,
            block_swap_ring_size=2,
            block_swap_granularity="block",
            use_pinned_memory_for_block_swap=self.config.use_pinned_memory_for_block_swap,
            lora_weights=(),
            lora_multipliers=(),
            compile_model=False,
            compile_backend="inductor",
            compile_mode="default",
            compile_dynamic=None,
            compile_fullgraph=False,
            compile_cache_size_limit=None,
            compile_auto_cache_size_limit=False,
            compile_fallback_to_eager=False,
            inductor_config=(),
            fused_qk_norm_rope=False,
            mode="fl2va",
        )

    def _conditioning(self, prompt: str) -> dict[str, torch.Tensor]:
        return self.conditioning[prompt]

    def _set_block_swap_training(self, training: bool) -> None:
        if not getattr(self.config, "blocks_to_swap", 0):
            return
        switch = self.transformer.switch_block_swap_for_training if training else self.transformer.switch_block_swap_for_inference
        switch()

    def _finish_block_swap_backward(self) -> None:
        if not getattr(self.config, "blocks_to_swap", 0):
            return
        finish = getattr(getattr(self.transformer, "offloader", None), "finish_truncated_backward", None)
        if callable(finish):
            finish(self.transformer.blocks)

    @contextmanager
    def _policy(self, state: dict[str, torch.Tensor] | None = None, *, enabled: bool = True) -> Iterator[None]:
        live = _cpu_state_dict(self.network) if state is not None else None
        was_enabled = self.network.is_enabled()
        try:
            if state is not None:
                self.network.load_state_dict(state, strict=True)
            self.network.set_enabled(enabled)
            yield
        finally:
            self.network.set_enabled(was_enabled)
            if live is not None:
                self.network.load_state_dict(live, strict=True)

    def _frame_count(self) -> int:
        return temporal_shape(round(self.config.duration * 24), align=True).frame_count

    @torch.no_grad()
    def _sample_latents(self, prompt: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        return denoise_fl2va(
            self.transformer,
            self._conditioning(prompt),
            height=self.config.height,
            width=self.config.width,
            frame_count=self._frame_count(),
            num_inference_steps=self.config.inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            condition_seed=seed,
            show_progress=False,
        )

    @torch.no_grad()
    def _save_media(self, video: torch.Tensor, audio: torch.Tensor, output: Path, prompt: str, seed: int) -> None:
        video_decoder = load_video_vae_decoder(self.config.video_vae, "cpu")
        audio_decoder = load_audio_vae_decoder(self.config.audio_vae, "cpu")
        media = decode_latents_sequentially(video_decoder, audio_decoder, video.cpu(), audio.cpu(), self.device)
        save_av_mp4(media, output, {"prompt": prompt, "seed": seed, "trainer": "h3-va-judger-nft"})

    def generate_group(
        self, prompt: str, *, virtual_step: int | None = None, output_dir: Path | None = None
    ) -> list[RolloutCandidate]:
        result: list[RolloutCandidate] = []
        rollout_step = self.step if virtual_step is None else virtual_step
        target_dir = self.rollout_dir if output_dir is None else output_dir
        self._set_block_swap_training(False)
        with self._policy(self.old_policy):
            for index in range(self.config.group_size):
                seed = self.config.seed + rollout_step * self.config.group_size + index
                video, audio = self._sample_latents(prompt, seed)
                path = target_dir / f"step_{rollout_step:06d}_{index:02d}.mp4"
                self._save_media(video, audio, path, prompt, seed)
                torch.save(
                    {"video": video.cpu(), "audio": audio.cpu(), "prompt": prompt, "seed": seed}, path.with_suffix(".latents.pt")
                )
                result.append(RolloutCandidate(prompt, seed, video.cpu(), audio.cpu(), path.resolve()))
        return result

    def _predict_x0(
        self,
        conditioning: dict[str, torch.Tensor],
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        noise_video: torch.Tensor,
        noise_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.transformer.config
        patch_size = tuple(config.patch_size)
        noisy_video = (1.0 - video_sigma) * clean_video + video_sigma * noise_video
        noisy_audio = (1.0 - audio_sigma) * clean_audio + audio_sigma * noise_audio
        text_hidden = conditioning[H3_TEXT_HIDDEN_KEY]
        text_tags = conditioning[H3_TEXT_TOKEN_TAGS_KEY]
        layout = build_t2va_packed_sequence(
            text_tags,
            num_latent_frames=clean_video.shape[2],
            latent_height=clean_video.shape[3],
            latent_width=clean_video.shape[4],
            num_audio_latents=clean_audio.shape[-1],
            patch_size=patch_size,
        )
        video_t = 1.0 - video_sigma
        audio_t = 1.0 - audio_sigma
        timestep, indices = build_row_timesteps(layout, video_t, audio_t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.transformer(
                video_hidden_states=patchify_video_latents(noisy_video, patch_size),
                audio_hidden_states=pack_audio_latents(noisy_audio),
                encoder_hidden_states=text_hidden[None].to(self.device),
                timestep=timestep.to(self.device),
                timestep_indices=indices.to(self.device),
                token_tags=layout.token_tags.to(self.device),
                token_tags_have_padding=bool((layout.token_tags < 0).any()),
                position_ids=layout.position_ids.to(self.device),
                video_indices=layout.video_indices.to(self.device),
                audio_indices=layout.audio_indices.to(self.device),
                text_indices=layout.text_indices.to(self.device),
            )
        video_velocity = unpatchify_video_tokens(output.video, latent_shape=clean_video.shape[1:], patch_size=patch_size)
        audio_velocity = unpack_audio_tokens(output.audio, num_audio_latents=clean_audio.shape[-1])
        return noisy_video + video_sigma * video_velocity.float(), noisy_audio + audio_sigma * audio_velocity.float()

    def _candidate_loss(self, candidate: RolloutCandidate, advantage: torch.Tensor) -> torch.Tensor:
        clean_video = candidate.video_latents.to(self.device, dtype=torch.float32)
        clean_audio = candidate.audio_latents.to(self.device, dtype=torch.float32)
        noise_video = torch.randn_like(clean_video)
        noise_audio = torch.randn_like(clean_audio)
        video_sigmas, _ = shifted_flow_schedule(self.config.inference_steps, VIDEO_FLOW_SHIFT, self.device)
        audio_sigmas, _ = shifted_flow_schedule(self.config.inference_steps, AUDIO_FLOW_SHIFT, self.device)
        index = int(torch.randint(0, min(video_sigmas.numel(), audio_sigmas.numel()) - 1, ()).item())
        video_sigma, audio_sigma = video_sigmas[index], audio_sigmas[index]
        cond = self._conditioning(candidate.prompt)
        self._set_block_swap_training(False)
        with torch.no_grad(), self._policy(self.old_policy):
            old_video, old_audio = self._predict_x0(
                cond, clean_video, clean_audio, noise_video, noise_audio, video_sigma, audio_sigma
            )
        self._set_block_swap_training(False)
        with torch.no_grad(), self._policy(enabled=False):
            base_video, base_audio = self._predict_x0(
                cond, clean_video, clean_audio, noise_video, noise_audio, video_sigma, audio_sigma
            )
        self._set_block_swap_training(True)
        current_video, current_audio = self._predict_x0(
            cond, clean_video, clean_audio, noise_video, noise_audio, video_sigma, audio_sigma
        )
        kwargs = dict(
            advantage=advantage.to(self.device),
            beta_mix=self.config.beta_mix,
            kl_beta=self.config.kl_beta,
            advantage_clip=self.config.advantage_clip,
        )
        video_loss = nft_mixture_loss(current_video, old_video, base_video, clean_video, **kwargs)
        audio_loss = nft_mixture_loss(current_audio, old_audio, base_audio, clean_audio, **kwargs)
        total_weight = self.config.video_loss_weight + self.config.audio_loss_weight
        return (self.config.video_loss_weight * video_loss + self.config.audio_loss_weight * audio_loss) / total_weight

    def update_group(
        self, candidates: list[RolloutCandidate], scores: torch.Tensor, *, refresh_old: bool = True
    ) -> dict[str, float]:
        if scores.shape != (len(candidates), 5) or not bool(torch.isfinite(scores).all()):
            raise ValueError(f"scores must be a finite {len(candidates)}x5 tensor")
        prompt = candidates[0].prompt
        components = reward_components(scores)
        advantages = dimension_normalized_advantages(components).cpu()
        reward_record = {
            "step": self.step,
            "prompt": prompt,
            "candidates": [{"path": str(candidate.media_path), "seed": candidate.seed} for candidate in candidates],
            "score_dimensions": ["A", "B", "C", "D", "E"],
            "raw_scores": scores.tolist(),
            "component_dimensions": ["overall", "audio", "video"],
            "components": components.tolist(),
            "advantages": advantages.tolist(),
        }
        reward_path = self.config.output_dir / f"step_{self.step:06d}.reward.json"
        reward_tmp = reward_path.with_name(reward_path.name + ".tmp")
        reward_tmp.write_text(json.dumps(reward_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reward_tmp.replace(reward_path)
        self.optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for index, candidate in enumerate(candidates):
            candidate_loss = self._candidate_loss(candidate, advantages[index]) / len(candidates)
            if not bool(torch.isfinite(candidate_loss)):
                raise FloatingPointError(f"non-finite NFT loss at step {self.step}, candidate {index}")
            try:
                candidate_loss.backward()
            finally:
                self._finish_block_swap_backward()
            detached_losses.append(candidate_loss.detach())
        nonfinite_gradients = [
            name
            for name, parameter in self.network.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite_gradients:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite adapter gradients: " + ", ".join(nonfinite_gradients[:8]))
        loss = torch.stack(detached_losses).sum()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.step += 1
        self.prompt_cursor += 1
        if refresh_old:
            self.old_policy = _cpu_state_dict(self.network)
        metrics = {
            "loss": float(loss.detach()),
            "reward_mean": float(scores.mean()),
            "reward_std": float(scores.std(unbiased=False)),
            "advantage_std": float(advantages.std(unbiased=False)),
        }
        with (self.config.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": self.step, **metrics}) + "\n")
        return metrics

    def train_step(self, prompt: str) -> dict[str, float]:
        if self.reward is None:
            raise RuntimeError("online train_step requires reward_endpoint")
        candidates = self.generate_group(prompt)
        for candidate in candidates:
            if not hasattr(candidate, "prompt"):
                candidate.prompt = prompt
        scores = self.reward.score_group(prompt, [candidate.media_path for candidate in candidates])
        return self.update_group(candidates, scores)

    def generate_round(self, round_dir: Path, group_count: int) -> Path:
        if group_count < 1:
            raise ValueError("group_count must be positive")
        group_count = min(group_count, self.config.max_steps - self.step)
        if group_count < 1:
            raise ValueError("no training groups remain before max_steps")
        round_dir = round_dir.resolve()
        if round_dir.exists():
            raise FileExistsError(f"round directory already exists: {round_dir}")
        round_dir.parent.mkdir(parents=True, exist_ok=True)
        working_dir = Path(tempfile.mkdtemp(prefix=f".{round_dir.name}.incomplete-", dir=round_dir.parent))
        groups = []
        start_step = self.step
        start_cursor = self.prompt_cursor
        behavior_sha256 = _policy_sha256(self.old_policy)
        for offset in range(group_count):
            virtual_step = start_step + offset
            prompt = self.prompts[(start_cursor + offset) % len(self.prompts)]
            candidates = self.generate_group(prompt, virtual_step=virtual_step, output_dir=working_dir)
            groups.append(
                {
                    "group_id": virtual_step,
                    "prompt": prompt,
                    "candidates": [
                        {
                            "path": str(round_dir / candidate.media_path.resolve().relative_to(working_dir)),
                            "latent_path": str(
                                round_dir / candidate.media_path.with_suffix(".latents.pt").resolve().relative_to(working_dir)
                            ),
                            "seed": candidate.seed,
                            "media_sha256": _file_sha256(candidate.media_path),
                            "latent_sha256": _file_sha256(candidate.media_path.with_suffix(".latents.pt")),
                        }
                        for candidate in candidates
                    ],
                }
            )
        resume_path = working_dir / "behavior.resume.pt"
        self._save_resume(resume_path)
        manifest = {
            "version": 1,
            "round_id": f"round-{start_step:06d}",
            "start_step": start_step,
            "prompt_cursor": start_cursor,
            "group_count": group_count,
            "behavior_sha256": behavior_sha256,
            "semantic_fingerprint": self.semantic_fingerprint,
            "resume_path": str(round_dir / "behavior.resume.pt"),
            "resume_sha256": _file_sha256(resume_path),
            "groups": groups,
        }
        manifest_path = working_dir / "manifest.json"
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_tmp.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        manifest_tmp.replace(manifest_path)
        working_dir.rename(round_dir)
        return round_dir / "manifest.json"

    def train_round(self, manifest_path: Path, rewards_path: Path) -> list[dict[str, float]]:
        manifest_path = manifest_path.resolve()
        rewards_path = rewards_path.resolve()
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        rewards = json.loads(rewards_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or rewards.get("version") != 1:
            raise ValueError("unsupported staged round contract version")
        judge = rewards.get("judge")
        if not isinstance(judge, dict) or not isinstance(judge.get("reward_model"), str) or not judge["reward_model"].strip():
            raise ValueError("reward metadata must identify the judge model")
        from musubi_tuner.minimax_h3_score_va_judger import (
            SCORING_IMPLEMENTATION_VERSION,
            SCORING_PROMPT_SHA256,
            SCORING_PROMPT_VERSION,
        )

        if (
            judge.get("backend") != "native"
            or judge.get("scoring_prompt_version") != SCORING_PROMPT_VERSION
            or judge.get("scoring_prompt_sha256") != SCORING_PROMPT_SHA256
            or judge.get("implementation_version") != SCORING_IMPLEMENTATION_VERSION
        ):
            raise ValueError("reward metadata does not match the built-in native scoring contract")
        for key in ("round_id", "behavior_sha256", "semantic_fingerprint"):
            if rewards.get(key) != manifest.get(key):
                raise ValueError(f"reward {key} does not match manifest")
        if rewards.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("reward manifest_sha256 does not match exact manifest bytes")
        if manifest.get("semantic_fingerprint") != self.semantic_fingerprint:
            raise ValueError("manifest changes training semantics")
        if manifest.get("start_step") != self.step or manifest.get("prompt_cursor") != self.prompt_cursor:
            raise ValueError("runtime is not at the manifest starting state")
        groups = manifest.get("groups")
        if not isinstance(groups, list) or not groups or manifest.get("group_count", 0) < 1:
            raise ValueError("manifest must contain at least one group")
        if manifest.get("group_count") != len(groups):
            raise ValueError("manifest group_count is inconsistent")
        if self.step + len(groups) > self.config.max_steps:
            raise ValueError("staged round exceeds configured max_steps")
        resume_path = Path(manifest.get("resume_path", ""))
        if not resume_path.is_file() or _file_sha256(resume_path) != manifest.get("resume_sha256"):
            raise ValueError("manifest behavior resume integrity check failed")
        behavior_hash = _policy_sha256(self.old_policy)
        if manifest.get("behavior_sha256") != behavior_hash or _policy_sha256(_cpu_state_dict(self.network)) != behavior_hash:
            raise ValueError("runtime policy does not match manifest behavior")
        marker = manifest_path.parent / "training.json"
        if (manifest_path.parent / "trained.json").exists():
            raise FileExistsError("staged round has already completed training")
        reward_groups = rewards.get("groups")
        if not isinstance(reward_groups, list) or len(reward_groups) != len(manifest["groups"]):
            raise ValueError("reward groups are incomplete")
        prepared: list[tuple[list[RolloutCandidate], torch.Tensor]] = []
        for offset, (group, judged) in enumerate(zip(groups, reward_groups, strict=True)):
            expected_group_id = self.step + offset
            expected_prompt = self.prompts[(self.prompt_cursor + offset) % len(self.prompts)]
            if group.get("group_id") != expected_group_id or group.get("prompt") != expected_prompt:
                raise ValueError("manifest group sequence does not match runtime prompts")
            if judged.get("group_id") != group.get("group_id") or judged.get("prompt") != group.get("prompt"):
                raise ValueError("reward group identity does not match manifest")
            entries = group.get("candidates", [])
            if not isinstance(entries, list) or len(entries) != self.config.group_size:
                raise ValueError("manifest candidate count does not match group_size")
            for index, entry in enumerate(entries):
                if entry.get("seed") != self.config.seed + expected_group_id * self.config.group_size + index:
                    raise ValueError("manifest candidate seed does not match rollout sequence")
            expected_hashes = [entry.get("media_sha256") for entry in entries]
            if judged.get("candidate_hashes") != expected_hashes:
                raise ValueError("reward candidate hashes do not match manifest")
            raw_scores = torch.as_tensor(judged.get("raw_scores"), dtype=torch.float32)
            if (
                raw_scores.shape != (len(entries), 5)
                or not bool(torch.isfinite(raw_scores).all())
                or not bool(((raw_scores >= 1) & (raw_scores <= 10)).all())
            ):
                raise ValueError("reward scores must be complete finite Kx5 values in [1, 10]")
            candidates = []
            for entry in entries:
                media_path, latent_path = Path(entry["path"]), Path(entry["latent_path"])
                if _file_sha256(media_path) != entry["media_sha256"] or _file_sha256(latent_path) != entry["latent_sha256"]:
                    raise ValueError("cached rollout integrity check failed")
                latent = torch.load(latent_path, map_location="cpu", weights_only=False)
                if latent.get("prompt") != group["prompt"] or int(latent.get("seed", -1)) != int(entry["seed"]):
                    raise ValueError("cached latent identity does not match manifest")
                if not isinstance(latent.get("video"), torch.Tensor) or not isinstance(latent.get("audio"), torch.Tensor):
                    raise ValueError("cached latent payload is incomplete")
                if (
                    latent["video"].numel() == 0
                    or latent["audio"].numel() == 0
                    or not bool(torch.isfinite(latent["video"]).all())
                    or not bool(torch.isfinite(latent["audio"]).all())
                ):
                    raise ValueError("cached latent tensors must be non-empty and finite")
                candidates.append(
                    RolloutCandidate(group["prompt"], int(entry["seed"]), latent["video"], latent["audio"], media_path)
                )
            prepared.append((candidates, raw_scores))
        marker_tmp = marker.with_name(marker.name + ".tmp")
        marker_tmp.write_text(
            json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "start_step": self.step}) + "\n",
            encoding="utf-8",
        )
        marker_tmp.replace(marker)
        metrics = [self.update_group(candidates, scores, refresh_old=False) for candidates, scores in prepared]
        self.old_policy = _cpu_state_dict(self.network)
        return metrics

    def save(self, name: str | None = None) -> tuple[Path, Path]:
        stem = name or f"step-{self.step:06d}"
        adapter = self.config.output_dir / f"{stem}.safetensors"
        resume = self.config.output_dir / f"{stem}.resume.pt"
        adapter_tmp = adapter.with_name(adapter.stem + ".tmp.safetensors")
        self.network.save_weights(
            str(adapter_tmp), torch.bfloat16, {"ss_training_type": "h3_va_judger_nft", "ss_step": str(self.step)}
        )
        adapter_tmp.replace(adapter)
        self._save_resume(resume)
        return adapter, resume

    def _save_resume(self, resume: Path) -> None:
        resume_tmp = resume.with_name(resume.name + ".tmp")
        torch.save(
            {
                "version": 1,
                "step": self.step,
                "prompt_cursor": self.prompt_cursor,
                "network": _cpu_state_dict(self.network),
                "old_policy": self.old_policy,
                "optimizer": self.optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.config).items()},
                "prompt_fingerprint": self.prompt_fingerprint,
                "semantic_config": self.semantic_config,
                "semantic_fingerprint": self.semantic_fingerprint,
            },
            resume_tmp,
        )
        resume_tmp.replace(resume)

    def load_resume(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != 1:
            raise ValueError(f"unsupported VA-Judger resume version in {path}")
        if state.get("prompt_fingerprint") != self.prompt_fingerprint:
            raise ValueError(f"prompt file contents differ from the resume {path}")
        saved_semantic = state.get("semantic_config")
        saved_fingerprint = state.get("semantic_fingerprint")
        if not isinstance(saved_semantic, dict) or not isinstance(saved_fingerprint, str):
            raise ValueError(f"resume {path} lacks semantic configuration metadata")
        if _config_fingerprint(saved_semantic) != saved_fingerprint:
            raise ValueError(f"resume {path} has corrupt semantic configuration metadata")
        comparable_saved = {key: value for key, value in saved_semantic.items() if key not in _RESUME_RUNTIME_KEYS}
        comparable_fingerprint = _config_fingerprint(comparable_saved)
        if comparable_fingerprint != self.semantic_fingerprint:
            changed = sorted(
                key
                for key in set(comparable_saved) | set(self.semantic_config)
                if comparable_saved.get(key) != self.semantic_config.get(key)
            )
            raise ValueError(f"resume {path} changes training semantics: {', '.join(changed)}")
        self.network.load_state_dict(state["network"], strict=True)
        self.old_policy = state["old_policy"]
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.step = int(state["step"])
        self.prompt_cursor = int(state["prompt_cursor"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    def run(self) -> None:
        while self.step < self.config.max_steps:
            prompt = self.prompts[self.prompt_cursor % len(self.prompts)]
            metrics = self.train_step(prompt)
            logger.info("step %d/%d prompt=%r metrics=%s", self.step, self.config.max_steps, prompt, metrics)
            if self.step % self.config.save_every == 0:
                self.save()
        self.save("final")
