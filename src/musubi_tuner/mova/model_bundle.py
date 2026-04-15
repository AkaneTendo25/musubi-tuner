from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

from .interaction import DualTowerConditionalBridge
from .wan_audio_dit import WanAudioModel
from .wan_video_dit import WanModel, sinusoidal_embedding_1d


class MovaModelBundle(nn.Module):
    def __init__(
        self,
        video_dit: WanModel,
        audio_dit: WanAudioModel,
        dual_tower_bridge: DualTowerConditionalBridge,
        alternate_video_dit: Optional[WanModel] = None,
        condition_scale: Optional[float] = None,
    ):
        super().__init__()
        self.video_dit = video_dit
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        if alternate_video_dit is not None:
            self.video_dit_2 = alternate_video_dit

        self.condition_scale = condition_scale
        self.use_gradient_checkpointing = False
        self.gradient_checkpointing_cpu_offload = False
        self.blocks_to_swap = 0
        self.visual_offloaders: dict[int, ModelOffloader] = {}
        self.audio_offloader: Optional[ModelOffloader] = None
        self.bridge_offloaders: dict[str, ModelOffloader] = {}
        self.bridge_block_index_maps: dict[str, dict[int, int]] = {}

    @property
    def dtype(self) -> torch.dtype:
        return next(self.video_dit.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.video_dit.parameters()).device

    def iter_visual_towers(self) -> list[WanModel]:
        towers = [self.video_dit]
        if hasattr(self, "video_dit_2"):
            towers.append(self.video_dit_2)
        return towers

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.use_gradient_checkpointing = True
        self.gradient_checkpointing_cpu_offload = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False
        self.gradient_checkpointing_cpu_offload = False

    def _maybe_checkpoint(self, fn, *args):
        if not (self.training and self.use_gradient_checkpointing):
            return fn(*args)
        if self.gradient_checkpointing_cpu_offload:
            with torch.autograd.graph.save_on_cpu():
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def _optimize_module_to_fp8(
        self,
        module: nn.Module,
        device: torch.device,
        move_to_device: bool,
        target_layer_keys: Optional[list[str]] = None,
        exclude_layer_keys: Optional[list[str]] = None,
        use_scaled_mm: bool = False,
    ):
        state_dict = module.state_dict()
        optimize_state_dict_with_fp8(
            state_dict,
            device,
            target_layer_keys=target_layer_keys,
            exclude_layer_keys=exclude_layer_keys,
            move_to_device=move_to_device,
        )
        apply_fp8_monkey_patch(module, state_dict, use_scaled_mm=use_scaled_mm)
        module.load_state_dict(state_dict, strict=True, assign=True)

    def fp8_optimization(self, device: torch.device, move_to_device: bool, use_scaled_mm: bool = False):
        for tower in self.iter_visual_towers():
            state_dict = tower.fp8_optimization(tower.state_dict(), device, move_to_device, use_scaled_mm=use_scaled_mm)
            tower.load_state_dict(state_dict, strict=True, assign=True)

        self._optimize_module_to_fp8(
            self.audio_dit,
            device,
            move_to_device,
            target_layer_keys=["blocks"],
            exclude_layer_keys=["norm", "modulation"],
            use_scaled_mm=use_scaled_mm,
        )
        self._optimize_module_to_fp8(
            self.dual_tower_bridge,
            device,
            move_to_device,
            target_layer_keys=["audio_to_video_conditioners", "video_to_audio_conditioners"],
            exclude_layer_keys=["norm", "probe", "condition_scale", "rotary", "emb"],
            use_scaled_mm=use_scaled_mm,
        )

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = blocks_to_swap
        self.visual_offloaders = {}
        self.audio_offloader = None
        self.bridge_offloaders = {}
        self.bridge_block_index_maps = {}
        if blocks_to_swap <= 0:
            return

        audio_blocks_to_swap = 0
        visual_blocks_to_swap = blocks_to_swap
        if len(self.audio_dit.blocks) > 1 and blocks_to_swap > 1:
            audio_blocks_to_swap = min(blocks_to_swap // 2, len(self.audio_dit.blocks) - 1)
            visual_blocks_to_swap = max(1, blocks_to_swap - audio_blocks_to_swap)

        for tower in self.iter_visual_towers():
            num_blocks = len(tower.blocks)
            if visual_blocks_to_swap > num_blocks - 1:
                raise ValueError(f"Cannot swap more than {num_blocks - 1} blocks for MOVA visual tower.")
            offloader = ModelOffloader(
                "mova_visual",
                tower.blocks,
                num_blocks,
                visual_blocks_to_swap,
                supports_backward,
                device,
                use_pinned_memory,
            )
            self.visual_offloaders[id(tower)] = offloader

        if audio_blocks_to_swap > 0:
            self.audio_offloader = ModelOffloader(
                "mova_audio",
                self.audio_dit.blocks,
                len(self.audio_dit.blocks),
                audio_blocks_to_swap,
                supports_backward,
                device,
                use_pinned_memory,
            )

        for direction, block_type in [("a2v", "mova_bridge_a2v"), ("v2a", "mova_bridge_v2a")]:
            bridge_blocks = self.dual_tower_bridge.get_conditioner_blocks(direction)
            if len(bridge_blocks) <= 1:
                continue
            bridge_blocks_to_swap = min(blocks_to_swap, len(bridge_blocks) - 1)
            if bridge_blocks_to_swap <= 0:
                continue
            self.bridge_offloaders[direction] = ModelOffloader(
                block_type,
                bridge_blocks,
                len(bridge_blocks),
                bridge_blocks_to_swap,
                supports_backward,
                device,
                use_pinned_memory,
            )
            self.bridge_block_index_maps[direction] = {
                layer_idx: block_idx
                for block_idx, layer_idx in enumerate(self.dual_tower_bridge.get_conditioner_layers(direction))
            }

    def switch_block_swap_for_inference(self):
        for offloader in self.visual_offloaders.values():
            offloader.set_forward_only(True)
        if self.audio_offloader is not None:
            self.audio_offloader.set_forward_only(True)
        for offloader in self.bridge_offloaders.values():
            offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        for offloader in self.visual_offloaders.values():
            offloader.set_forward_only(False)
        if self.audio_offloader is not None:
            self.audio_offloader.set_forward_only(False)
        for offloader in self.bridge_offloaders.values():
            offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device):
        for tower in self.iter_visual_towers():
            if id(tower) not in self.visual_offloaders:
                tower.to(device)
                continue

            saved_blocks = tower.blocks
            tower.blocks = None
            tower.to(device)
            tower.blocks = saved_blocks

        if self.audio_offloader is None:
            self.audio_dit.to(device)
        else:
            saved_audio_blocks = self.audio_dit.blocks
            self.audio_dit.blocks = None
            self.audio_dit.to(device)
            self.audio_dit.blocks = saved_audio_blocks

        bridge_a2v_offloader = self.bridge_offloaders.get("a2v")
        bridge_v2a_offloader = self.bridge_offloaders.get("v2a")
        if bridge_a2v_offloader is None and bridge_v2a_offloader is None:
            self.dual_tower_bridge.to(device)
        else:
            saved_a2v = self.dual_tower_bridge.audio_to_video_conditioners
            saved_v2a = self.dual_tower_bridge.video_to_audio_conditioners
            if bridge_a2v_offloader is not None:
                self.dual_tower_bridge.audio_to_video_conditioners = None
            if bridge_v2a_offloader is not None:
                self.dual_tower_bridge.video_to_audio_conditioners = None
            self.dual_tower_bridge.to(device)
            self.dual_tower_bridge.audio_to_video_conditioners = saved_a2v
            self.dual_tower_bridge.video_to_audio_conditioners = saved_v2a

    def prepare_block_swap_before_forward(self):
        for tower in self.iter_visual_towers():
            offloader = self.visual_offloaders.get(id(tower))
            if offloader is not None:
                offloader.prepare_block_devices_before_forward(tower.blocks)
        if self.audio_offloader is not None:
            self.audio_offloader.prepare_block_devices_before_forward(self.audio_dit.blocks)
        if "a2v" in self.bridge_offloaders:
            self.bridge_offloaders["a2v"].prepare_block_devices_before_forward(
                self.dual_tower_bridge.get_conditioner_blocks("a2v")
            )
        if "v2a" in self.bridge_offloaders:
            self.bridge_offloaders["v2a"].prepare_block_devices_before_forward(
                self.dual_tower_bridge.get_conditioner_blocks("v2a")
            )

    def _tower_offloader(self, tower: WanModel) -> Optional[ModelOffloader]:
        return self.visual_offloaders.get(id(tower))

    def _bridge_offloader(self, direction: str) -> Optional[ModelOffloader]:
        return self.bridge_offloaders.get(direction)

    def _bridge_block_index(self, layer_idx: int, direction: str) -> Optional[int]:
        return self.bridge_block_index_maps.get(direction, {}).get(layer_idx)

    def forward_model(
        self,
        *,
        visual_dit: WanModel,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        conditioning_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: Optional[torch.Tensor] = None,
        video_fps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = video_latents.device
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
            fps_value = float(video_fps[0].item()) if video_fps is not None else 16.0
            cross_visual_freqs, cross_audio_freqs = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=fps_value,
                grid_size=visual_grid,
                audio_steps=audio_grid[0],
                device=device,
                dtype=model_dtype,
            )

        visual_offloader = self._tower_offloader(visual_dit)
        audio_offloader = self.audio_offloader
        min_layers = min(len(visual_dit.blocks), len(audio_dit.blocks))
        for layer_idx in range(min_layers):
            if visual_offloader is not None:
                visual_offloader.wait_for_block(layer_idx)
            if audio_offloader is not None:
                audio_offloader.wait_for_block(layer_idx)

            a2v_block_idx = None
            v2a_block_idx = None
            a2v_offloader = self._bridge_offloader("a2v")
            v2a_offloader = self._bridge_offloader("v2a")
            if a2v_offloader is not None and self.dual_tower_bridge.should_interact(layer_idx, "a2v"):
                a2v_block_idx = self._bridge_block_index(layer_idx, "a2v")
                if a2v_block_idx is not None:
                    a2v_offloader.wait_for_block(a2v_block_idx)
            if v2a_offloader is not None and self.dual_tower_bridge.should_interact(layer_idx, "v2a"):
                v2a_block_idx = self._bridge_block_index(layer_idx, "v2a")
                if v2a_block_idx is not None:
                    v2a_offloader.wait_for_block(v2a_block_idx)

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
                if a2v_offloader is not None and a2v_block_idx is not None:
                    a2v_offloader.submit_move_blocks_forward(self.dual_tower_bridge.get_conditioner_blocks("a2v"), a2v_block_idx)
                if v2a_offloader is not None and v2a_block_idx is not None:
                    v2a_offloader.submit_move_blocks_forward(self.dual_tower_bridge.get_conditioner_blocks("v2a"), v2a_block_idx)

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

            if visual_offloader is not None:
                visual_offloader.submit_move_blocks_forward(visual_dit.blocks, layer_idx)
            if audio_offloader is not None:
                audio_offloader.submit_move_blocks_forward(audio_dit.blocks, layer_idx)

        for offset, block in enumerate(visual_dit.blocks[min_layers:], start=min_layers):
            if visual_offloader is not None:
                visual_offloader.wait_for_block(offset)
            visual_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=block: block(x, ctx, t_mod, freqs),
                visual_x,
                visual_context,
                visual_t_mod.to(dtype=model_dtype),
                visual_freqs,
            )
            if visual_offloader is not None:
                visual_offloader.submit_move_blocks_forward(visual_dit.blocks, offset)

        for offset, block in enumerate(audio_dit.blocks[min_layers:], start=min_layers):
            if audio_offloader is not None:
                audio_offloader.wait_for_block(offset)
            audio_x = self._maybe_checkpoint(
                lambda x, ctx, t_mod, freqs, block=block: block(x, ctx, t_mod, freqs),
                audio_x,
                audio_context,
                audio_t_mod.to(dtype=model_dtype),
                audio_freqs,
            )
            if audio_offloader is not None:
                audio_offloader.submit_move_blocks_forward(audio_dit.blocks, offset)

        video_pred = visual_dit.unpatchify(visual_dit.head(visual_x, visual_t.to(dtype=model_dtype)), visual_grid)
        audio_pred = audio_dit.unpatchify(audio_dit.head(audio_x, audio_t.to(dtype=model_dtype)), audio_grid)
        return video_pred, audio_pred
