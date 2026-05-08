import torch
from musubi_tuner.ltx_2.components.patchifiers import get_pixel_coords
from musubi_tuner.ltx_2.conditioning.item import ConditioningItem
from musubi_tuner.ltx_2.tools import VideoLatentTools
from musubi_tuner.ltx_2.types import LatentState, VideoLatentShape


class VideoConditionByKeyframeIndex(ConditioningItem):
    """
    Conditions video generation on keyframe latents at a specific frame index.
    Appends keyframe tokens to the latent state with positions offset by frame_idx,
    and sets denoise strength according to the strength parameter.

    Args:
        keyframes: Keyframe latents [B, C, F, H, W].
        frame_idx: Frame index offset for positional encoding.
        strength: Conditioning strength (1.0 = clean, 0.0 = fully denoised).
        num_pixel_frames: Number of pixel frames the keyframe latent originally
            encodes. When set to 1, the temporal end position of each token is
            narrowed to start+1 (single-pixel-frame extent) instead of the
            VAE-scaled range. Default 1 matches upstream's typical use where
            keyframes represent single-pixel-frame anchors.
    """

    def __init__(
        self,
        keyframes: torch.Tensor,
        frame_idx: int,
        strength: float,
        num_pixel_frames: int = 1,
    ):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength
        self.num_pixel_frames = num_pixel_frames

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.keyframes.shape),
            device=self.keyframes.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        )

        positions[:, 0, ...] += self.frame_idx
        if self.num_pixel_frames == 1:
            positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
        )
