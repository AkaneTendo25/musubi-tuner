from enum import Enum
from typing import Protocol

import logging
import os
import torch
from musubi_tuner.ltx_2.model.transformer.rope import LTXRopeType, apply_rotary_emb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

memory_efficient_attention = None
flash_attn_interface = None
flash_attention_2 = None
flash_attn_varlen_func = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    from flash_attn import flash_attn_func as flash_attention_2
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
    flash_attn_varlen_func = _flash_attn_varlen_func
except ImportError:
    flash_attention_2 = None
    flash_attn_varlen_func = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention2(AttentionCallable):
    def _pack_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        # mask is attention bias or boolean mask; normalize to [B, Lk] boolean valid mask
        if mask.dtype == torch.bool:
            valid = mask
        else:
            valid = mask == 0
        while valid.dim() > 2:
            valid = valid.squeeze(1)

        bsz, q_len, heads, dim_head = q.shape
        k_len = k.shape[1]
        if valid.shape[1] != k_len:
            raise ValueError(f"FlashAttention2 varlen expects mask length {k_len}, got {valid.shape[1]}")

        seqlens_k = valid.sum(dim=1).to(dtype=torch.int32)
        max_seqlen_k = int(seqlens_k.max().item()) if seqlens_k.numel() else k_len
        if max_seqlen_k == 0:
            raise ValueError("FlashAttention2 varlen received an all-masked context.")

        q_packed = q.reshape(bsz * q_len, heads, dim_head)
        cu_seqlens_q = torch.arange(
            0,
            (bsz + 1) * q_len,
            step=q_len,
            device=q.device,
            dtype=torch.int32,
        )

        k_list = [k[i, valid[i]] for i in range(bsz)]
        v_list = [v[i, valid[i]] for i in range(bsz)]
        k_packed = torch.cat(k_list, dim=0)
        v_packed = torch.cat(v_list, dim=0)

        cu_seqlens_k = torch.zeros((bsz + 1,), device=k.device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

        return (
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q,
            cu_seqlens_k,
            q_len,
            max_seqlen_k,
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attention_2 is None:
            raise RuntimeError("FlashAttention2 was selected but `flash_attn` is not installed.")
        if mask is not None and flash_attn_varlen_func is None:
            logger.warning("FlashAttention2 does not support attention masks; falling back to PyTorch SDPA.")
            return PytorchAttention()(q, k, v, heads, mask)

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        if mask is not None:
            q_packed, k_packed, v_packed, cu_q, cu_k, max_q, max_k = self._pack_varlen(q, k, v, mask)
            out = flash_attn_varlen_func(q_packed, k_packed, v_packed, cu_q, cu_k, max_q, max_k)
            out = out.view(b, max_q, heads, dim_head)
        else:
            out = flash_attention_2(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_2:
            return FlashAttention2()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        split_attn_mode: str | None = None,
        split_attn_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function
        self.split_attn_mode = split_attn_mode
        self.split_attn_chunk_size = split_attn_chunk_size

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        from musubi_tuner.ltx_2.utils import RMSNorm
        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def _slice_mask_for_batch(self, mask: torch.Tensor | None, idx: int, batch_size: int) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape[0] == batch_size:
            return mask[idx : idx + 1]
        return mask

    def _slice_mask_for_query(
        self, mask: torch.Tensor | None, start: int, end: int, q_len: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 4 and mask.shape[-2] == q_len:
            return mask[..., start:end, :]
        if mask.ndim == 3 and mask.shape[1] == q_len:
            return mask[:, start:end, :]
        if mask.ndim == 2 and mask.shape[0] == q_len:
            return mask[start:end, :]
        return mask

    def _run_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.attention_function(q, k, v, self.heads, mask)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if os.getenv("LTX2_FP8_DEVICE_DIAG", "0") == "1":
            try:
                org_fwd = getattr(self.to_q, "org_forward", None)
                orig = getattr(org_fwd, "__self__", None) if callable(org_fwd) else None
                w = getattr(orig, "weight", None)
                sw = getattr(orig, "scale_weight", None)
                if isinstance(w, torch.Tensor) and isinstance(sw, torch.Tensor):
                    if w.device != sw.device or w.device != x.device:
                        logger.error(
                            "LTX-2 fp8 device diag: to_q weight=%s scale=%s input=%s",
                            w.device,
                            sw.device,
                            x.device,
                        )
            except Exception:
                pass
        q = self.to_q(x)
        if os.getenv("LTX2_FP8_DEVICE_DIAG", "0") == "1":
            try:
                w = getattr(self.to_q, "weight", None)
                sw = getattr(self.to_q, "scale_weight", None)
                if isinstance(w, torch.Tensor) and isinstance(sw, torch.Tensor):
                    if w.device != sw.device or w.device != x.device:
                        logger.error(
                            "LTX-2 fp8 device diag: to_q weight=%s scale=%s input=%s",
                            w.device,
                            sw.device,
                            x.device,
                        )
            except Exception:
                pass
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        split_mode = (self.split_attn_mode or "").lower()
        if split_mode == "batch":
            outs = []
            for i in range(q.shape[0]):
                mask_i = self._slice_mask_for_batch(mask, i, q.shape[0])
                out_i = self._run_attention(q[i : i + 1], k[i : i + 1], v[i : i + 1], mask_i)
                outs.append(out_i)
            out = torch.cat(outs, dim=0)
        elif split_mode == "query":
            chunk_size = int(self.split_attn_chunk_size) if self.split_attn_chunk_size else 1024
            if chunk_size <= 0:
                chunk_size = 1024
            outs = []
            q_len = q.shape[1]
            for start in range(0, q_len, chunk_size):
                end = min(start + chunk_size, q_len)
                mask_chunk = self._slice_mask_for_query(mask, start, end, q_len)
                out_chunk = self._run_attention(q[:, start:end], k, v, mask_chunk)
                outs.append(out_chunk)
            out = torch.cat(outs, dim=1)
        else:
            # attention_function can be an enum *or* a custom callable
            out = self._run_attention(q, k, v, mask)
        return self.to_out(out)
