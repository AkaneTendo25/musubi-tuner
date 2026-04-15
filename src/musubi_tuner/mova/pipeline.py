from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .interaction import DualTowerConditionalBridge
from .wan_audio_dit import WanAudioModel
from .wan_video_dit import WanModel, sinusoidal_embedding_1d


class MovaLoRATrainingPipeline(nn.Module):
    def __init__(
        self,
        visual_dit: WanModel,
        audio_dit: WanAudioModel,
        dual_tower_bridge: DualTowerConditionalBridge,
        *,
        audio_loss_weight: float = 1.0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        condition_scale: Optional[float] = None,
    ):
        super().__init__()
        self.visual_dit = visual_dit
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        self.audio_loss_weight = audio_loss_weight
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.condition_scale = condition_scale

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.visual_dit.parameters()).dtype

    def _maybe_checkpoint(self, fn, *args):
        if not (self.training and self.use_gradient_checkpointing):
            return fn(*args)
        if self.use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def forward_model(
        self,
        *,
        visual_dit: Optional[WanModel] = None,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        conditioning_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: Optional[torch.Tensor] = None,
        video_fps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = video_latents.device
        visual_dit = visual_dit if visual_dit is not None else self.visual_dit
        model_dtype = next(visual_dit.parameters()).dtype
        audio_dit = self.audio_dit

        if audio_timesteps is None:
            audio_timesteps = timesteps

        visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timesteps))
        visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))
        audio_t = audio_dit.time_embedding(sinusoidal_embedding_1d(audio_dit.freq_dim, audio_timesteps))
        audio_t_mod = audio_dit.time_projection(audio_t).unflatten(1, (6, audio_dit.dim))

        visual_context = visual_dit.text_embedding(prompt_embeds)
        audio_context = audio_dit.text_embedding(prompt_embeds)

        visual_x = video_latents.to(dtype=model_dtype)
        if visual_dit.require_vae_embedding:
            if conditioning_latents is None:
                raise ValueError("conditioning_latents are required for this MOVA visual tower")
            visual_x = torch.cat([visual_x, conditioning_latents.to(dtype=model_dtype)], dim=1)
        audio_x = audio_latents.to(dtype=model_dtype)

        visual_x, visual_grid = visual_dit.patchify(visual_x)
        audio_x, audio_grid = audio_dit.patchify(audio_x)
        visual_freqs = visual_dit.build_freqs(visual_grid, device=device)
        audio_freqs = audio_dit.build_freqs(audio_grid, device=device)

        cross_visual_freqs = None
        cross_audio_freqs = None
        if self.dual_tower_bridge.apply_cross_rope:
            fps_value = float(video_fps[0].item()) if video_fps is not None else 24.0
            cross_visual_freqs, cross_audio_freqs = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=fps_value,
                grid_size=visual_grid,
                audio_steps=audio_grid[0],
                device=device,
                dtype=model_dtype,
            )

        min_layers = min(len(visual_dit.blocks), len(audio_dit.blocks))
        for layer_idx in range(min_layers):
            if self.dual_tower_bridge.should_interact(layer_idx, "a2v") or self.dual_tower_bridge.should_interact(layer_idx, "v2a"):
                def bridge_fn(layer_arg, visual_arg, audio_arg):
                    return self.dual_tower_bridge(
                        int(layer_arg.item()),
                        visual_arg,
                        audio_arg,
                        x_freqs=cross_visual_freqs,
                        y_freqs=cross_audio_freqs,
                        condition_scale=self.condition_scale,
                        video_grid_size=visual_grid,
                    )

                layer_tensor = torch.tensor(layer_idx, device=device)
                visual_x, audio_x = self._maybe_checkpoint(bridge_fn, layer_tensor, visual_x, audio_x)

            visual_block = visual_dit.blocks[layer_idx]
            audio_block = audio_dit.blocks[layer_idx]
            visual_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=visual_block: block(x, ctx, t_mod, freqs),
                visual_x,
                visual_context,
                visual_t_mod.to(dtype=model_dtype),
                visual_freqs,
            )
            audio_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=audio_block: block(x, ctx, t_mod, freqs),
                audio_x,
                audio_context,
                audio_t_mod.to(dtype=model_dtype),
                audio_freqs,
            )

        for block in visual_dit.blocks[min_layers:]:
            visual_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=block: block(x, ctx, t_mod, freqs),
                visual_x,
                visual_context,
                visual_t_mod.to(dtype=model_dtype),
                visual_freqs,
            )

        for block in audio_dit.blocks[min_layers:]:
            audio_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=block: block(x, ctx, t_mod, freqs),
                audio_x,
                audio_context,
                audio_t_mod.to(dtype=model_dtype),
                audio_freqs,
            )

        video_pred = visual_dit.unpatchify(visual_dit.head(visual_x, visual_t.to(dtype=model_dtype)), visual_grid)
        audio_pred = audio_dit.unpatchify(audio_dit.head(audio_x, audio_t.to(dtype=model_dtype)), audio_grid)
        return video_pred, audio_pred

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        visual_dit: Optional[WanModel] = None,
        min_timestep_boundary: float = 0.0,
        max_timestep_boundary: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        device = batch["video_latents"].device
        visual_dit = visual_dit if visual_dit is not None else self.visual_dit
        model_dtype = next(visual_dit.parameters()).dtype

        video_latents = batch["video_latents"].to(device=device, dtype=model_dtype)
        audio_latents = batch["audio_latents"].to(device=device, dtype=model_dtype)
        conditioning_latents = batch["conditioning_latents"].to(device=device, dtype=model_dtype)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=model_dtype)
        video_fps = batch["video_fps"].to(device=device, dtype=torch.float32)

        timesteps = torch.rand(video_latents.shape[0], device=device, dtype=torch.float32)
        timesteps = min_timestep_boundary + timesteps * (max_timestep_boundary - min_timestep_boundary)
        model_timesteps = timesteps * 1000.0 + 1.0

        video_noise = torch.randn_like(video_latents)
        audio_noise = torch.randn_like(audio_latents)
        video_sigma = timesteps.view(-1, 1, 1, 1, 1).to(dtype=model_dtype)
        audio_sigma = timesteps.view(-1, 1, 1).to(dtype=model_dtype)
        noisy_video = (1.0 - video_sigma) * video_latents + video_sigma * video_noise
        noisy_audio = (1.0 - audio_sigma) * audio_latents + audio_sigma * audio_noise

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=model_dtype)
            if device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_ctx:
            video_pred, audio_pred = self.forward_model(
                visual_dit=visual_dit,
                video_latents=noisy_video,
                audio_latents=noisy_audio,
                conditioning_latents=conditioning_latents,
                prompt_embeds=prompt_embeds,
                timesteps=model_timesteps,
                audio_timesteps=model_timesteps,
                video_fps=video_fps,
            )

        video_target = video_noise - video_latents
        audio_target = audio_noise - audio_latents
        video_loss = torch.nn.functional.mse_loss(video_pred.float(), video_target.float())
        audio_loss = torch.nn.functional.mse_loss(audio_pred.float(), audio_target.float())
        loss = video_loss + self.audio_loss_weight * audio_loss
        return {
            "loss": loss,
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "timestep": model_timesteps.detach().mean(),
        }
