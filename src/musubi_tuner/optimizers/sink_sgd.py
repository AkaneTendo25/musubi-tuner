"""SinkSGD optimizer.

Adapted for Musubi Tuner from ``adv_optm`` 2.4.dev17
(``SinkSGD_adv``), which is Apache-2.0 licensed.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn.functional as F

from musubi_tuner.ltx2_sinksgd_defaults import (
    DEFAULT_LTX2_SINKSGD_MOMENTUM,
    DEFAULT_LTX2_SINKSGD_NESTEROV,
    DEFAULT_LTX2_SINKSGD_NESTEROV_COEF,
    DEFAULT_LTX2_SINKSGD_NORMED_MOMENTUM,
    DEFAULT_LTX2_SINKSGD_ORTHOGONAL_SINKHORN,
    DEFAULT_LTX2_SINKSGD_SINKHORN_ITERATIONS,
)


_generators: dict[torch.device, torch.Generator] = {}
_INT8_STATE_BLOCK_SIZE = 2048


def _set_seed(device: torch.device) -> None:
    if device not in _generators:
        _generators[device] = torch.Generator(device=device)
    _generators[device].manual_seed(42)


def _get_generator(device: torch.device) -> torch.Generator:
    if device not in _generators:
        _set_seed(device)
    return _generators[device]


def _copy_stochastic_core_(target: torch.Tensor, source: torch.Tensor, random_int_tensor: torch.Tensor) -> None:
    result = random_int_tensor.clone()
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def _copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    random_int = _get_random_int_for_sr(source)
    _copy_stochastic_core_(target, source, random_int)


def _get_random_int_for_sr(source: torch.Tensor) -> torch.Tensor:
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=_get_generator(source.device),
    )


def _get_random_int_for_fp8_sr(source: torch.Tensor) -> torch.Tensor:
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 20),
        generator=_get_generator(source.device),
    )


def _get_random_int_for_8bit_sr(source: torch.Tensor, numel: int) -> torch.Tensor:
    return torch.randint(
        size=(numel,),
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=_get_generator(source.device),
    )


def _get_random_noise_for_sign(source: torch.Tensor) -> torch.Tensor:
    return torch.rand(
        source.shape,
        device=source.device,
        dtype=source.dtype,
        generator=_get_generator(source.device),
    ).mul_(2).sub_(1)


def _prepare_int8_blocks(value: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    orig_shape = tuple(value.shape)
    orig_numel = value.numel()
    pad_len = (block_size - (orig_numel % block_size)) % block_size
    flat = value.reshape(-1)
    if pad_len:
        if orig_numel == 0:
            flat = torch.zeros(pad_len, device=value.device, dtype=value.dtype)
        else:
            flat = torch.cat((flat, flat[-1:].expand(pad_len)))
    return flat.view(-1, block_size).float(), orig_shape, orig_numel


def _init_state_tensor(
    state: dict[str, Any],
    key: str,
    shape: torch.Size,
    state_precision: str,
    device: torch.device,
    default_dtype: torch.dtype,
) -> None:
    if state_precision == "fp32":
        store_dtype = torch.float32
    elif state_precision == "bf16_sr":
        store_dtype = torch.bfloat16
    elif state_precision == "fp8_sr":
        store_dtype = torch.float8_e4m3fn
    elif state_precision == "int8_sr":
        store_dtype = torch.int8
    else:
        store_dtype = default_dtype

    if store_dtype == getattr(torch, "float8_e4m3fn", None):
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.tensor(1.0, device=device, dtype=torch.float32)
    elif store_dtype == torch.int8:
        n_blocks = (math.prod(shape) + _INT8_STATE_BLOCK_SIZE - 1) // _INT8_STATE_BLOCK_SIZE
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.ones(n_blocks, device=device, dtype=torch.float32)
    else:
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)


def _get_state(state: dict[str, Any], key: str, state_precision: str) -> torch.Tensor:
    tensor = state[key]
    if state_precision == "fp8_sr":
        return tensor.float() / state[f"{key}_scale"]
    if state_precision == "int8_sr":
        scales = state[f"{key}_scale"]
        blocks, orig_shape, orig_numel = _prepare_int8_blocks(tensor, _INT8_STATE_BLOCK_SIZE)
        return (blocks * scales.unsqueeze(1)).view(-1)[:orig_numel].view(orig_shape)
    if state_precision == "bf16_sr":
        return tensor.float()
    return tensor


def _copy_fp8_stochastic_(
    target: torch.Tensor,
    source: torch.Tensor,
    scale: torch.Tensor,
    random_int: torch.Tensor | None = None,
) -> None:
    if random_int is None:
        random_int = _get_random_int_for_fp8_sr(source)
    buffer = (source * scale).to(torch.float32)
    sign = torch.sign(buffer)
    buffer.abs_()
    offset = (buffer < 0.015625).to(torch.float32).mul_(0.015625)
    buffer.add_(offset)
    buffer_int = buffer.view(torch.int32)
    buffer_int.add_(random_int)
    buffer_int.bitwise_and_(-1048576)
    buffer = buffer_int.view(torch.float32)
    buffer.sub_(offset)
    buffer.mul_(sign)
    target.copy_(buffer.to(torch.float8_e4m3fn))


def _copy_int8_sym_blockwise_stochastic_(
    target: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    random_int: torch.Tensor | None = None,
    block_size: int = _INT8_STATE_BLOCK_SIZE,
    blocks: torch.Tensor | None = None,
) -> None:
    if blocks is None:
        blocks, orig_shape, orig_numel = _prepare_int8_blocks(source, block_size)
    else:
        orig_shape = tuple(source.shape)
        orig_numel = source.numel()
    if random_int is None:
        random_int = _get_random_int_for_8bit_sr(source, blocks.numel())
    normalised = blocks / scales.float().clamp_min(1e-12).unsqueeze(1)
    noise = random_int.reshape_as(blocks).float().mul_(1.0 / (1 << 16))
    quantised = normalised.add_(noise).floor_().clamp_(-127, 127).to(torch.int8)
    target.copy_(quantised.view(-1)[:orig_numel].view(orig_shape))


def _set_state(
    state: dict[str, Any],
    key: str,
    value: torch.Tensor,
    state_precision: str,
    random_int: torch.Tensor | None = None,
) -> None:
    if state_precision == "fp32":
        if state[key] is not value:
            state[key].copy_(value)
    elif state_precision == "fp8_sr":
        amax = value.abs().max().clamp_min(1e-12)
        scale = 448.0 / amax
        state[f"{key}_scale"].copy_(scale)
        _copy_fp8_stochastic_(state[key], value, scale, random_int)
    elif state_precision == "bf16_sr":
        if random_int is None:
            _copy_stochastic_(state[key], value)
        else:
            _copy_stochastic_core_(state[key], value, random_int)
    elif state_precision == "int8_sr":
        blocks, _, _ = _prepare_int8_blocks(value, _INT8_STATE_BLOCK_SIZE)
        scales = blocks.abs().amax(dim=1).div_(127.0)
        state[f"{key}_scale"].copy_(scales)
        _copy_int8_sym_blockwise_stochastic_(state[key], value, scales, random_int, blocks=blocks)
    else:
        if state[key] is not value:
            state[key].copy_(value)


def _upcast_grad_for_precision(grad: torch.Tensor, state: dict[str, Any], state_precision: str) -> torch.Tensor:
    if state.get("factored", False) or state_precision in {"bf16_sr", "fp8_sr", "int8_sr", "factored"}:
        return grad.float()
    return grad


def _get_effective_shape(numel: int) -> tuple[int, int]:
    if numel <= 0:
        return (0, 0)
    for dim in reversed(range(1, int(math.sqrt(numel)) + 1)):
        if numel % dim == 0:
            return numel // dim, dim
    return numel, 1


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    bits = (signs > 0).to(torch.uint8)
    rows, cols = bits.shape
    padded_cols = ((cols + 7) // 8) * 8
    if padded_cols != cols:
        bits = F.pad(bits, (0, padded_cols - cols))
    bits = bits.view(rows, -1, 8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return (bits << shifts).sum(dim=-1).to(torch.uint8)


def _unpack_signs(packed: torch.Tensor, cols: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[0], -1)[:, :cols]
    return bits.to(torch.float32).mul_(2.0).sub_(1.0)


def _factorize_state(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sign = _pack_signs(value > 0)
    abs_value = value.abs().float()
    rows, cols = abs_value.shape
    row = torch.sum(abs_value, dim=1, dtype=torch.float32)
    col = torch.sum(abs_value, dim=0, dtype=torch.float32)
    eps = 1e-12
    if rows < cols:
        row.div_(row.sum().clamp_min_(eps))
    else:
        col.div_(col.sum().clamp_min_(eps))
    return row, col, sign


def _reconstruct_state(row: torch.Tensor, col: torch.Tensor, packed_sign: torch.Tensor, cols: int) -> torch.Tensor:
    return torch.outer(row.float(), col.float()) * _unpack_signs(packed_sign, cols)


def _is_wd_centered(p: torch.Tensor) -> bool:
    if getattr(p, "_is_lora_A", False) or getattr(p, "_is_lora_B", False):
        return True
    if getattr(p, "_is_oft", False):
        return True
    if getattr(p, "_is_dora_scale", False):
        return False
    return False


def _quantize_anchor_blockwise(p: torch.Tensor, block_size: int, bits: int = 8):
    val_flat = p.flatten()
    pad_len = (block_size - (val_flat.numel() % block_size)) % block_size
    val_padded = torch.cat((val_flat, val_flat[-1:].expand(pad_len))) if pad_len else val_flat
    blocks = val_padded.view(-1, block_size).float()
    min_vals, max_vals = torch.aminmax(blocks, dim=1, keepdim=True)
    max_int = (1 << bits) - 1
    scales = (max_vals - min_vals).div_(float(max_int))
    scales.masked_fill_(scales == 0, 1.0)
    quantized = blocks.sub_(min_vals).div_(scales).round_().clamp_(0, max_int).to(torch.uint8)
    return quantized, scales.squeeze(1), min_vals.squeeze(1)


def _init_anchor(p: torch.Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
    if not group.get("centered_wd", 0.0) or _is_wd_centered(p) or "anchor_data" in state:
        return

    mode = group.get("centered_wd_mode", "full")
    numel = p.numel()
    if numel == 0 or (mode in {"int8", "int4"} and numel < 10000) or p.ndim == 1 or getattr(p, "_is_dora_scale", False):
        state["anchor_data"] = p.detach().clone()
    elif mode == "float8":
        state["anchor_data"] = p.detach().to(torch.float8_e4m3fn)
    elif mode == "int8":
        q_blocks, scales, mins = _quantize_anchor_blockwise(p.detach(), block_size=128, bits=8)
        state["anchor_data"] = q_blocks
        state["anchor_scale"] = scales.to(p.dtype)
        state["anchor_min"] = mins.to(p.dtype)
    elif mode == "int4":
        q_blocks, scales, mins = _quantize_anchor_blockwise(p.detach(), block_size=32, bits=4)
        q_flat = q_blocks.view(-1)
        if q_flat.numel() % 2 == 1:
            q_flat = F.pad(q_flat, (0, 1))
        state["anchor_data"] = (q_flat[0::2] << 4) | q_flat[1::2]
        state["anchor_scale"] = scales.to(p.dtype)
        state["anchor_min"] = mins.to(p.dtype)
    elif mode == "full":
        state["anchor_data"] = p.detach().clone()
    else:
        raise ValueError(f"centered_wd_mode must be one of full, float8, int8, int4. Got {mode}")


def _dequantize_anchor(p: torch.Tensor, state: dict[str, Any], group: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    anchor_data = state["anchor_data"]
    if anchor_data.dtype in (p.dtype, torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return anchor_data.to(dtype)

    mode = group.get("centered_wd_mode", "full")
    scales = state["anchor_scale"]
    mins = state["anchor_min"]
    if mode == "int4" and anchor_data.dtype == torch.uint8:
        block_size = 32
        unpacked = torch.empty(anchor_data.numel() * 2, dtype=torch.uint8, device=anchor_data.device)
        unpacked[0::2] = anchor_data >> 4
        unpacked[1::2] = anchor_data & 0x0F
        quantized_blocks = unpacked.view(-1, block_size)
    elif mode == "int8" and anchor_data.dtype == torch.uint8:
        quantized_blocks = anchor_data
    else:
        return anchor_data.to(dtype)

    anchor_blocks = quantized_blocks.float() * scales.float().unsqueeze(1) + mins.float().unsqueeze(1)
    return anchor_blocks.view(-1)[: p.numel()].view(p.shape).to(dtype)


def _apply_sr_sinkhorn(update: torch.Tensor, p: torch.Tensor, ortho_project: bool, iters: int = 5) -> torch.Tensor:
    original_shape = update.shape
    original_dtype = update.dtype
    update = update.float()

    if update.dim() == 1:
        if ortho_project:
            p_float = p.float()
            p_norm_sq = torch.dot(p_float, p_float).add_(1e-30)
            update.sub_(p_float * (torch.dot(p_float, update) / p_norm_sq))
        norm = update.norm(p=2).clamp_min_(1e-12)
        return update.mul_(math.sqrt(update.numel()) / norm).view(original_shape).to(original_dtype)

    update_2d = update.view(update.shape[0], -1)
    m, n = update_2d.shape
    scale_cond = m > n
    dim = 0 if scale_cond else 1
    scale_first = math.sqrt(m if scale_cond else n)
    scale_second = math.sqrt(n if scale_cond else m)

    if ortho_project:
        param_2d = p.float().view(p.shape[0], -1)
        p_norm_sq_dim = torch.sum(param_2d * param_2d, dim=dim, keepdim=True).add_(1e-30)
        p_norm_sq_adim = torch.sum(param_2d * param_2d, dim=1 - dim, keepdim=True).add_(1e-30)

    for _ in range(iters):
        norm1 = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_first / norm1)
        if ortho_project:
            update_2d = _ortho_normed(param_2d, update_2d, p_norm_sq_dim, dim, scale_first)

        norm2 = update_2d.norm(p=2, dim=1 - dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_second / norm2)
        if ortho_project:
            update_2d = _ortho_normed(param_2d, update_2d, p_norm_sq_adim, 1 - dim, scale_second)

    return update_2d.view(original_shape).to(original_dtype)


def _ortho_normed(
    p_2d: torch.Tensor,
    update_2d: torch.Tensor,
    p_norm_sq: torch.Tensor,
    dim: int,
    target_norm: float,
) -> torch.Tensor:
    dot_prod = torch.sum(p_2d * update_2d, dim=dim, keepdim=True)
    update_2d.addcmul_(dot_prod / p_norm_sq, p_2d, value=-1.0)
    g_orth_norm = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update_2d.mul_(target_norm / g_orth_norm)


def _apply_stochastic_sign_(update: torch.Tensor, noise: torch.Tensor | None, is_vector: bool = False) -> torch.Tensor:
    if update.dim() >= 2 and not is_vector:
        row_norm = torch.linalg.vector_norm(update, ord=float("inf"), dim=1, keepdim=True).clamp_min_(1e-12)
        update.div_(row_norm)
        col_norm = torch.linalg.vector_norm(update, ord=float("inf"), dim=0, keepdim=True).clamp_min_(1e-12)
        update.div_(col_norm)
    else:
        block_size = 128
        numel = update.numel()
        if numel <= block_size:
            update.div_(update.abs().max().clamp_min_(1e-12))
        else:
            flat = update.reshape(-1)
            remainder = numel % block_size
            padded = F.pad(flat, (0, block_size - remainder)) if remainder else flat
            blocks = padded.view(-1, block_size)
            scale = blocks.abs().max(dim=1, keepdim=True).values
            update.div_(scale.expand_as(blocks).reshape(-1)[:numel].view_as(update).clamp_min(1e-12))

    if noise is None:
        noise = _get_random_noise_for_sign(update)
    return update.add_(noise).sign_()


def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    original_shape = grad.shape
    original_dtype = grad.dtype
    p_flat = p.view(-1).float()
    grad_flat = grad.view(-1).float()
    p_norm_sq = torch.dot(p_flat, p_flat).add_(1e-30)
    grad_orth = grad_flat.sub(p_flat * (torch.dot(p_flat, grad_flat) / p_norm_sq))
    grad_norm = grad_flat.norm(2)
    grad_orth_norm = grad_orth.norm(2).add_(1e-30)
    return (grad_orth * (grad_norm / grad_orth_norm)).view(original_shape).to(original_dtype)


def _is_spectral(p: torch.Tensor) -> bool:
    if p.ndim < 2:
        return False
    if getattr(p, "_is_oft", False) or getattr(p, "_is_dora_scale", False) or getattr(p, "is_vector", False):
        return False
    return getattr(p, "is_hidden", True)


def _init_spectral_norm(state: dict[str, Any], p: torch.Tensor) -> None:
    generator = _get_generator(p.device)
    d_out = p.shape[0]
    d_in = p.numel() // d_out
    v = torch.randn(d_in, device=p.device, dtype=p.dtype, generator=generator)
    u = torch.randn(d_out, device=p.device, dtype=p.dtype, generator=generator)
    state["spectral_v"] = v.div_(v.norm().add_(1e-12))
    state["spectral_u"] = u.div_(u.norm().add_(1e-12))


def _rms_normalization(update: torch.Tensor, dim: int | None, lr: float | torch.Tensor) -> torch.Tensor:
    n = update.numel() if dim is None else update.shape[dim]
    norm_eps = 1 / math.sqrt(n)
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min(norm_eps)
    return update.mul_(lr * math.sqrt(n) / norm)


def _max_row_norm_normalization(update: torch.Tensor, lr: float | torch.Tensor, target_scale: float = 0.5) -> torch.Tensor:
    max_norm = torch.linalg.vector_norm(update, ord=2, dim=1).max().clamp_min(1.0 / math.sqrt(update.shape[1]))
    return update.mul_(lr * target_scale / max_norm)


def _spectral_normalization(
    update: torch.Tensor,
    u_state: torch.Tensor,
    v_state: torch.Tensor,
    lr: float | torch.Tensor,
    target_scale: float,
) -> torch.Tensor:
    d_out = update.shape[0]
    d_in = update.numel() // d_out
    update = update.to(u_state.dtype)
    update_flat = update.view(d_out, d_in)

    v_raw = torch.mv(update_flat.mT, u_state)
    v_norm = torch.linalg.vector_norm(v_raw)
    v_state.copy_(torch.where(v_norm >= 1e-6, v_raw / v_norm.clamp_min(1e-8), v_state))

    u_raw = torch.mv(update_flat, v_state)
    u_norm = torch.linalg.vector_norm(u_raw)
    u_state.copy_(torch.where(u_norm >= 1e-6, u_raw / u_norm.clamp_min(1e-8), u_state))

    sigma = torch.linalg.vecdot(u_state, u_raw)
    spectral_eps = 1.0 / (math.sqrt(d_out) + math.sqrt(d_in))
    return update.mul_(lr * (target_scale / sigma.clamp_min_(spectral_eps)))


def _scale_update(p: torch.Tensor, update: torch.Tensor, lr: float | torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    if p.ndim < 2 or getattr(p, "_is_dora_scale", False):
        return _rms_normalization(update, dim=None, lr=lr)

    if getattr(p, "_is_oft", False):
        return _max_row_norm_normalization(update, lr)

    d_out = update.shape[0]
    d_in = update.numel() // d_out
    target_scale = 1.0 if getattr(p, "_is_lora_A", False) else math.sqrt(d_out / d_in)
    return _spectral_normalization(update, state["spectral_u"], state["spectral_v"], lr=lr, target_scale=target_scale)


def _adjust_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    if getattr(p, "_is_dora_scale", False):
        return 0.0, cwd
    if getattr(p, "_is_oft", False):
        return wd, 0.0
    if p.ndim >= 2:
        if getattr(p, "_is_lora_A", False) or getattr(p, "_is_lora_B", False):
            return wd, 0.0
        return wd, cwd
    return 0.0, cwd


def _apply_weight_decay(
    p_calc: torch.Tensor,
    update_calc: torch.Tensor,
    p: torch.Tensor,
    state: dict[str, Any],
    group: dict[str, Any],
    scaled_wd: float | torch.Tensor | None,
    scaled_cwd: float | torch.Tensor | None,
) -> None:
    cautious = group.get("cautious_wd", False)
    if scaled_wd is not None:
        if cautious:
            mask = (update_calc * p_calc >= 0).to(p_calc.dtype)
            if isinstance(scaled_wd, torch.Tensor):
                p_calc.addcmul_(p_calc, mask * scaled_wd, value=-1.0)
            else:
                p_calc.addcmul_(p_calc, mask, value=-scaled_wd)
        else:
            if isinstance(scaled_wd, torch.Tensor):
                p_calc.addcmul_(p_calc, scaled_wd, value=-1.0)
            else:
                p_calc.add_(p_calc, alpha=-scaled_wd)

    if scaled_cwd is not None and "anchor_data" in state:
        anchor = _dequantize_anchor(p, state, group, p_calc.dtype)
        decay_target = p_calc.sub(anchor)
        if cautious:
            mask = (update_calc * decay_target >= 0).to(p_calc.dtype)
            if isinstance(scaled_cwd, torch.Tensor):
                p_calc.addcmul_(decay_target, mask * scaled_cwd, value=-1.0)
            else:
                p_calc.addcmul_(decay_target, mask, value=-scaled_cwd)
        else:
            if isinstance(scaled_cwd, torch.Tensor):
                p_calc.addcmul_(decay_target, scaled_cwd, value=-1.0)
            else:
                p_calc.add_(decay_target, alpha=-scaled_cwd)


class SinkSGD(torch.optim.Optimizer):
    """SGD with Sinkhorn update normalization and optional spectral scaling."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = DEFAULT_LTX2_SINKSGD_MOMENTUM,
        weight_decay: float = 0.0,
        sinkhorn_iterations: int = DEFAULT_LTX2_SINKSGD_SINKHORN_ITERATIONS,
        orthogonal_sinkhorn: bool = DEFAULT_LTX2_SINKSGD_ORTHOGONAL_SINKHORN,
        normed_momentum: bool = DEFAULT_LTX2_SINKSGD_NORMED_MOMENTUM,
        nesterov: bool = DEFAULT_LTX2_SINKSGD_NESTEROV,
        nesterov_coef: float | None = DEFAULT_LTX2_SINKSGD_NESTEROV_COEF,
        decoupled_wd: bool = False,
        cautious_wd: bool = False,
        stochastic_rounding: bool = True,
        orthogonal_gradient: bool = False,
        spectral_normalization: bool = False,
        centered_wd: float = 0.0,
        centered_wd_mode: str = "float8",
        state_precision: str = "auto",
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        compiled_optimizer: bool = False,
    ) -> None:
        momentum = float(momentum)
        if nesterov_coef is not None:
            nesterov_coef = float(nesterov_coef)

        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0.0. Got {lr}")
        if momentum < 0.0:
            raise ValueError(f"Momentum should be >= 0.0. Got {momentum}")
        if momentum >= 1.0:
            raise ValueError(f"Momentum should be < 1.0. Got {momentum}")
        if nesterov_coef is not None and not 0.0 <= nesterov_coef <= 1.0:
            raise ValueError(f"nesterov_coef should be between 0.0 and 1.0. Got {nesterov_coef}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay should be >= 0.0. Got {weight_decay}")
        if sinkhorn_iterations < 1:
            raise ValueError(f"sinkhorn_iterations should be >= 1. Got {sinkhorn_iterations}")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp8_sr", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")
        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "nesterov_coef": nesterov_coef,
            "normed_momentum": normed_momentum,
            "decoupled_wd": decoupled_wd,
            "cautious_wd": cautious_wd,
            "orthogonal_gradient": orthogonal_gradient,
            "compiled_optimizer": compiled_optimizer,
            "sinkhorn_iterations": sinkhorn_iterations,
            "orthogonal_sinkhorn": orthogonal_sinkhorn,
            "spectral_normalization": spectral_normalization,
            "centered_wd": centered_wd,
            "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
        }
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group["params"] if p.dtype == torch.bfloat16}
            for device in devices:
                _set_seed(device)

        self._compiled_step_parameter = None
        self._compile_warning_emitted = False
        self.init_step()
        if compiled_optimizer:
            self.compile(fullgraph=True)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state:
                    self._fix_loaded_state_dtype(state, p, group)

    def _fix_loaded_state_dtype(self, state: dict[str, Any], p: torch.Tensor, group: dict[str, Any]) -> None:
        actual_precision = state.get("actual_state_precision", group.get("actual_state_precision", "auto"))
        if actual_precision == "fp32" or p.numel() < 10000 or p.ndim == 1:
            base_dtype = torch.float32
        elif actual_precision == "bf16_sr":
            base_dtype = torch.bfloat16
        elif actual_precision == "fp8_sr":
            base_dtype = torch.float8_e4m3fn
        elif actual_precision == "int8_sr":
            base_dtype = torch.float32
        else:
            base_dtype = torch.float32 if state.get("factored", False) else p.dtype

        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"sign"}:
                state[key] = value.to(device=p.device, dtype=torch.uint8)
            elif key.endswith("_scale") and key != "anchor_scale":
                state[key] = value.to(device=p.device, dtype=torch.float32)
            elif key in {"mu_b_nmf", "mv_b_nmf"}:
                state[key] = value.to(device=p.device, dtype=torch.float32)
            elif key == "anchor_data":
                if group.get("centered_wd_mode", "full") in {"int8", "int4"} and p.ndim != 1 and p.numel() >= 10000:
                    state[key] = value.to(device=p.device, dtype=torch.uint8)
                else:
                    state[key] = value.to(device=p.device)
            elif key in {"anchor_scale", "anchor_min"}:
                state[key] = value.to(device=p.device, dtype=p.dtype)
            elif key == "momentum_buffer" and actual_precision == "int8_sr":
                state[key] = value.to(device=p.device, dtype=torch.int8)
            elif value.is_floating_point():
                state[key] = value.to(device=p.device, dtype=base_dtype)
            else:
                state[key] = value.to(device=p.device)

    def init_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        state = self.state[p]
        if "step" in state:
            return

        state["step"] = 0
        req_precision = group["state_precision"]
        is_vector = len(p.shape) == 1 and not group["vector_reshape"]
        state["factored"] = req_precision == "factored" and not is_vector

        actual_precision = "auto" if req_precision == "factored" else req_precision
        if actual_precision != "auto" and (p.numel() < 10000 or p.ndim == 1):
            actual_precision = "fp32"
        state["actual_state_precision"] = actual_precision
        group["actual_state_precision"] = actual_precision

        dtype = torch.float32 if (state["factored"] or req_precision == "factored") else p.dtype
        if group["momentum"] != 0:
            if state["factored"]:
                state["effective_shape"] = _get_effective_shape(p.numel())
                d1, d2 = state["effective_shape"]
                state["mu_b_nmf"] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                state["mv_b_nmf"] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                state["sign"] = torch.zeros((d1, (d2 + 7) // 8), dtype=torch.uint8, device=p.device)
            else:
                _init_state_tensor(state, "momentum_buffer", p.shape, actual_precision, p.device, dtype)

        if group.get("spectral_normalization", False) and _is_spectral(p):
            _init_spectral_norm(state, p)

        _init_anchor(p, state, group)

    @torch.no_grad()
    def step_param(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        self.step_parameter(p, group)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict[str, Any], i: int | None = None) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        self.__init_state(p, group)
        actual_precision = state.get("actual_state_precision", group.get("actual_state_precision", "auto"))
        random_int_tensor = None
        random_int_state_tensor = None
        sign_noise = None

        use_compiled = bool(group.get("compiled_optimizer", False))
        if use_compiled:
            step_size = torch.as_tensor(group["lr"], device=p.device)
            is_vector = p.grad.ndim < 2 or getattr(p, "_is_dora_scale", False) or getattr(p, "is_vector", False)
            if is_vector:
                sign_noise = _get_random_noise_for_sign(p)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                random_int_tensor = _get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            if actual_precision == "bf16_sr" and random_int_state_tensor is None:
                random_int_state_tensor = _get_random_int_for_sr(p)
            elif actual_precision == "int8_sr":
                padded_numel = ((p.numel() + _INT8_STATE_BLOCK_SIZE - 1) // _INT8_STATE_BLOCK_SIZE) * _INT8_STATE_BLOCK_SIZE
                random_int_state_tensor = _get_random_int_for_8bit_sr(p, padded_numel)
            elif actual_precision == "fp8_sr":
                random_int_state_tensor = _get_random_int_for_fp8_sr(p)
        else:
            step_size = group["lr"]

        if use_compiled:
            try:
                if self._compiled_step_parameter is None:
                    self.compile(fullgraph=True)
                self._compiled_step_parameter(
                    p,
                    p.grad,
                    state,
                    group,
                    step_size,
                    random_int_tensor,
                    random_int_state_tensor,
                    sign_noise,
                )
            except Exception as exc:
                self._disable_compiled_optimizer(group, exc)
                self._step_parameter(p, p.grad, state, group, group["lr"])
        else:
            self._step_parameter(p, p.grad, state, group, step_size)
        state["step"] += 1

    def _step_parameter(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        step_size: float | torch.Tensor,
        random_int_tensor: torch.Tensor | None = None,
        random_int_state_tensor: torch.Tensor | None = None,
        sign_noise: torch.Tensor | None = None,
    ) -> None:
        grad = _upcast_grad_for_precision(grad, state, group["state_precision"])
        is_vector = grad.ndim < 2 or getattr(p, "_is_dora_scale", False) or getattr(p, "is_vector", False)
        sinkhorn_iterations = group["sinkhorn_iterations"]
        orthogonal_sinkhorn = group["orthogonal_sinkhorn"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        nesterov_coef = group.get("nesterov_coef", None)

        if group.get("normed_momentum", False):
            if not is_vector:
                grad = _apply_sr_sinkhorn(grad, p, ortho_project=orthogonal_sinkhorn, iters=sinkhorn_iterations)
            else:
                grad = _apply_stochastic_sign_(grad, sign_noise, is_vector=is_vector)

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        if state["factored"]:
            d1, d2 = state["effective_shape"]
            grad_reshaped = grad.view(d1, d2)
            if momentum != 0:
                buf = _reconstruct_state(state["mu_b_nmf"], state["mv_b_nmf"], state["sign"], d2)
                buf.lerp_(grad_reshaped, 1 - momentum)
                state["mu_b_nmf"], state["mv_b_nmf"], state["sign"] = _factorize_state(buf.clone())
                update = grad_reshaped.lerp(buf, momentum if nesterov_coef is None else nesterov_coef) if nesterov else buf.clone()
            else:
                update = grad_reshaped.clone()
            update = update.view(p.shape)
        else:
            actual_precision = state.get("actual_state_precision", group["actual_state_precision"])
            if momentum != 0:
                buf = _get_state(state, "momentum_buffer", actual_precision)
                buf.lerp_(grad, 1 - momentum)
                _set_state(state, "momentum_buffer", buf, actual_precision, random_int_state_tensor)
                update = grad.lerp(buf, momentum if nesterov_coef is None else nesterov_coef) if nesterov else buf.clone()
            else:
                update = grad.clone()

        if not group.get("normed_momentum", False):
            if not is_vector:
                update = _apply_sr_sinkhorn(update, p, ortho_project=orthogonal_sinkhorn, iters=sinkhorn_iterations)
            else:
                update = _apply_stochastic_sign_(update, sign_noise, is_vector=is_vector)

        if group.get("spectral_normalization", False):
            update = _scale_update(p, update, step_size, state=state)
        else:
            update.mul_(step_size)

        self._apply_parameter_update(p, group, update, step_size, random_int_tensor)

    def _apply_parameter_update(
        self,
        p: torch.Tensor,
        group: dict[str, Any],
        update: torch.Tensor,
        lr: float | torch.Tensor,
        random_int_tensor: torch.Tensor | None = None,
    ) -> None:
        wd, cwd = _adjust_wds(float(group["weight_decay"]), float(group.get("centered_wd", 0.0)), p)
        decay_factor = (lr / self._init_lr) if group.get("decoupled_wd", False) and self._init_lr != 0 else lr
        scaled_wd = wd * decay_factor if wd != 0 else None
        scaled_cwd = cwd * decay_factor if cwd != 0 else None
        state = self.state[p]

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            p_fp32 = p.float()
            update_fp32 = update.float()
            if scaled_wd is not None or scaled_cwd is not None:
                _apply_weight_decay(p_fp32, update_fp32, p, state, group, scaled_wd, scaled_cwd)
            p_fp32.add_(-update_fp32)
            if random_int_tensor is None:
                _copy_stochastic_(p, p_fp32)
            else:
                _copy_stochastic_core_(p, p_fp32, random_int_tensor)
        else:
            if scaled_wd is not None or scaled_cwd is not None:
                _apply_weight_decay(p, update, p, state, group, scaled_wd, scaled_cwd)
            p.add_(-update)

    def compile(self, *args, **kwargs) -> None:
        try:
            self._compiled_step_parameter = torch.compile(self._step_parameter, *args, **kwargs)
        except Exception as exc:
            self._disable_compiled_optimizer(None, exc)

    def _disable_compiled_optimizer(self, failed_group: dict[str, Any] | None, exc: Exception) -> None:
        self._compiled_step_parameter = None
        if failed_group is not None:
            failed_group["compiled_optimizer"] = False
        for group in self.param_groups:
            group["compiled_optimizer"] = False
        if not self._compile_warning_emitted:
            warnings.warn(
                "SinkSGD compiled_optimizer failed and was disabled for this run; continuing with eager SinkSGD. "
                f"Original error: {exc.__class__.__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compile_warning_emitted = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        return loss


SinkSGD_adv = SinkSGD
