"""Few-step π-Flow inference sampler for validation (LAY-396).

``piflow_generate`` replaces the Euler ``generate()`` denoise loop with the
π-Flow segment loop: ``nfe`` network calls, each building a network-free
``DXPolicy`` that is integrated over its segment by ``policy_rollout_fm``.

The sampler is integrated into the shared SR stage runner, so the SR validation
metrics can drive the grid DiT student without a duplicate generation wrapper.
"""

from __future__ import annotations

from typing import Any

import torch

# (inference adaptation) the generation helpers live in generation.utils here,
# and decode goes through vae_io so the kvae denormalization ((x+1)*128) is
# honoured.
from musubi_tuner.kandinsky6.runtime.sr.core.algo.piflow_math import DXPolicy, policy_rollout_fm, shift_timesteps
from musubi_tuner.kandinsky6.runtime.sr.core.algo.train_utils import get_sparse_params


def build_piflow_sampler_params(piflow_conf: Any, n_grid: int) -> dict[str, Any]:
    """Sampler kwargs for ``piflow_generate``, read from ``conf.trainer.piflow``.

    Attach the result to the DiT as ``dit.piflow_params`` so ``generate_sample_sr``
    dispatches to the DX sampler without monkeypatching the module symbol — this works
    for every caller (incl. ``sr_video_generation``) and both latent and pixel datasets.
    """
    return {
        "nfe": int(piflow_conf.nfe),
        "num_policy_substeps": int(piflow_conf.num_policy_substeps),
        "final_step_size_scale": float(piflow_conf.final_step_size_scale),
        "shift": float(getattr(piflow_conf, "shift", None) or 5.0),
        "n_grid": int(n_grid),
        "eps": float(piflow_conf.eps),
    }


@torch.no_grad()
def piflow_generate(  # noqa: PLR0913
    img: torch.Tensor,
    model: torch.nn.Module,
    text_embeds: dict[str, torch.Tensor],
    visual_cu_seqlens: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    visual_rope_pos: list[torch.Tensor],
    text_rope_pos: torch.Tensor,
    scale_factor: tuple[float, ...],
    *,
    nfe: int,
    num_policy_substeps: int,
    final_step_size_scale: float,
    shift: float,
    n_grid: int,
    out_dim: int,
    eps: float = 1e-6,
    tp_mesh: dict[str, Any] | None = None,
    start_timestep: float = 1.0,
    device: str | int | None = None,
    progress_callback=None,
    scheduler: Any | None = None,
) -> torch.Tensor:
    """Few-step π-Flow denoise: ``nfe`` network calls, network-free integration between.

    Walks the deterministic ``nfe``-segment schedule from ``raw_t = start_timestep``
    down to 0 (final segment scaled by ``final_step_size_scale``, matching ``sample_t``).
    Each segment: one ``model`` forward -> ``DXPolicy`` -> integrate over the segment.
    """
    device = device if device is not None else img.device

    img = img.to(device)
    visual_cu_seqlens = visual_cu_seqlens.to(device)
    text_cu_seqlens = text_cu_seqlens.to(device)
    visual_rope_pos = [position.to(device) for position in visual_rope_pos]
    text_rope_pos = text_rope_pos.to(device)
    text_embeds = {key: value.to(device) for key, value in text_embeds.items()}
    sparse_params = get_sparse_params(model, img, visual_cu_seqlens)
    if tp_mesh:
        tp_world_size = tp_mesh["tp"].size()
        tp_rank = tp_mesh["tp"].get_local_rank()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]

    ndim = img.dim()
    if nfe < 1:
        msg = f"piflow_generate requires nfe >= 1, got {nfe}"
        raise ValueError(msg)
    if scheduler is not None:
        scheduler.set_timesteps(nfe, device=device)
        if scheduler.timesteps.shape[0] != nfe:
            raise ValueError(f"Piflow scheduler returned {scheduler.timesteps.shape[0]} steps, expected {nfe}")
    else:
        # Mirror sample_t's guard so a tiny/zero final_step_size_scale can't zero the denominator.
        final_step_size_scale = max(float(final_step_size_scale), eps)
        one_minus_final = 1.0 - final_step_size_scale
        base_seg = 1.0 / (float(nfe) - one_minus_final)
        final_seg = final_step_size_scale * base_seg

    x = img
    for step_index in range(nfe):
        if scheduler is not None:
            timestep = scheduler.timesteps[step_index]
            sigma_src = scheduler.sigmas[step_index].item()
        else:
            idx = nfe - step_index
            raw_src = min(max((idx - one_minus_final) * base_seg, eps), start_timestep)
            seg = final_seg if idx == 1 else base_seg
            raw_dst = max(raw_src - seg, 0.0)
            sigma_src = shift_timesteps(torch.tensor(raw_src, device=device), shift).item()

        n_objects = visual_cu_seqlens.shape[0] - 1
        t_src = torch.full(
            (n_objects,),
            float(timestep.item()) if scheduler is not None else sigma_src * 1000.0,
            device=device,
            dtype=x.dtype,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v0 = model(
                x,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                t_src,
                visual_cu_seqlens,
                text_cu_seqlens,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=scale_factor,
                sparse_params=sparse_params,
            )
        if scheduler is not None:
            x_pred = scheduler.step(v0, timestep, x[..., :out_dim], return_dict=False)[0]
        else:
            v0_grid = v0.unsqueeze(1) if n_grid == 1 else v0
            total_tokens = x.shape[0]
            sigma_tok = torch.full((total_tokens, *((ndim - 1) * [1])), sigma_src, device=device)
            seg_tok = torch.full((total_tokens,), seg, device=device)
            policy = DXPolicy(v0_grid, x[..., :out_dim], sigma_tok, seg_tok, shift=shift, mode="grid", eps=eps)

            raw_src_tok = torch.full((total_tokens,), raw_src, device=device)
            raw_dst_tok = torch.full((total_tokens,), raw_dst, device=device)
            x_pred, _, _ = policy_rollout_fm(
                x[..., :out_dim], sigma_tok, raw_src_tok, raw_dst_tok, num_policy_substeps, policy
            )
        x = torch.cat([x_pred, x[..., out_dim:]], dim=-1)
        if progress_callback is not None:
            progress_callback()
    return x[..., :out_dim]
