import logging
import os
import torch
import torch.nn as nn

from typing import Iterable, List, Optional

from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import (
    ModelOffloader,
    _clean_memory_on_device,
    _synchronize_device,
    weighs_to_device,
    params_to_device,
)

logger = logging.getLogger(__name__)
_LOGGED_SWAP_BYTES = False
_LOGGED_FIRST_PARAM = False
_SKIP_AUDIO_SWAP = os.getenv("LTX2_SWAP_SKIP_AUDIO", "1") == "1"
_SKIP_CROSS_ATTN_SWAP = os.getenv("LTX2_SWAP_KEEP_CROSS_ATTN", "0") == "1"
_SKIP_ATTN_SWAP = os.getenv("LTX2_SWAP_KEEP_ATTN", "0") == "1"


def _should_skip_swap(name: str) -> bool:
    if _SKIP_AUDIO_SWAP and "audio" in name:
        return True
    if _SKIP_ATTN_SWAP and "attn" in name:
        return True
    if _SKIP_CROSS_ATTN_SWAP and ("audio_to_video_attn" in name or "video_to_audio_attn" in name):
        return True
    return False


def _move_non_linear_params(block: nn.Module, device: torch.device) -> None:
    """Move non-linear params/buffers to device; Linear weights are handled by offloader."""
    non_blocking = device.type != "cpu"
    for submodule in block.modules():
        if isinstance(submodule, nn.Linear):
            # Keep Linear weights on CPU; move bias/scale_weight if present.
            if submodule.bias is not None and submodule.bias.device != device:
                submodule.bias.data = submodule.bias.data.to(device, non_blocking=non_blocking)
            if hasattr(submodule, "scale_weight") and submodule.scale_weight is not None and submodule.scale_weight.device != device:
                submodule.scale_weight.data = submodule.scale_weight.data.to(device, non_blocking=non_blocking)
            continue
        for param in submodule.parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device, non_blocking=non_blocking)
        for buf in submodule.buffers(recurse=False):
            if buf.device != device:
                buf.data = buf.data.to(device, non_blocking=non_blocking)


def _move_block_params_excluding_audio(
    block: nn.Module,
    device: torch.device,
    *,
    include_norms: bool,
    use_pinned: bool,
) -> None:
    for name, module in block.named_modules():
        if _should_skip_swap(name):
            continue
        if include_norms:
            params_to_device(module, device, include_norms=True, use_pinned=use_pinned)
        else:
            if module.__class__.__name__.endswith("Linear"):
                weighs_to_device(module, device, use_pinned=use_pinned)


def _log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info("LTX-2 swap mem [%s]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB", tag, allocated, reserved)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _module_cuda_bytes(module: nn.Module) -> int:
    total = 0
    for tensor in module.parameters():
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            total += _tensor_bytes(tensor)
    for tensor in module.buffers():
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            total += _tensor_bytes(tensor)
    return total


def _log_block_cuda_bytes(blocks: List[nn.Module], split_idx: int) -> None:
    kept = blocks[:split_idx]
    swapped = blocks[split_idx:]
    kept_bytes = sum(_module_cuda_bytes(block) for block in kept)
    swapped_bytes = sum(_module_cuda_bytes(block) for block in swapped)
    logger.info(
        "LTX-2 swap mem [blocks_cuda_bytes]: kept=%.2fMB swapped=%.2fMB",
        kept_bytes / (1024**2),
        swapped_bytes / (1024**2),
    )


def _log_first_param(blocks: List[nn.Module]) -> None:
    for block in blocks:
        for param in block.parameters():
            logger.info(
                "LTX-2 swap diag [first_param]: dtype=%s device=%s",
                param.dtype,
                param.device,
            )
            return


def _summarize_block_tensors(block: nn.Module, label: str) -> None:
    entries = []
    for name, param in block.named_parameters(recurse=True):
        if isinstance(param, torch.Tensor):
            entries.append((name, param.device, _tensor_bytes(param)))
    for name, buf in block.named_buffers(recurse=True):
        if isinstance(buf, torch.Tensor):
            entries.append((f"{name} (buffer)", buf.device, _tensor_bytes(buf)))
    entries.sort(key=lambda item: item[2], reverse=True)
    for name, device, size in entries[:8]:
        logger.info("LTX-2 swap diag [%s]: %s device=%s size=%.2fMB", label, name, device, size / (1024**2))


def _module_on_device(module: nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for tensor in module.parameters():
        if tensor.device != target:
            return False
    for tensor in module.buffers():
        if tensor.device != target:
            return False
    return True


class LTX2BlockSwapManager:
    """Stream full blocks between devices for LTX-2."""

    def __init__(self, block_indices: List[int], offload_device: torch.device):
        self.block_indices = set(block_indices)
        self.offload_device = offload_device
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hook_device: Optional[torch.device] = None

    @classmethod
    def build(
        cls,
        depth: int,
        blocks_to_swap: int,
        swap_device: str,
    ) -> Optional["LTX2BlockSwapManager"]:
        if not blocks_to_swap:
            return None
        max_swappable_blocks = max(depth - 1, 0)
        if max_swappable_blocks == 0:
            return None
        if blocks_to_swap > max_swappable_blocks:
            logger.warning(
                "Requested LTX-2 aggressive blocks_to_swap=%s but maximum swappable blocks is %s; clamping.",
                blocks_to_swap,
                max_swappable_blocks,
            )
            blocks_to_swap = max_swappable_blocks
        try:
            offload_device = torch.device(swap_device)
        except Exception as exc:
            logger.warning(
                "Failed to initialize LTX-2 aggressive block swap; continuing without offload: %s",
                exc,
            )
            return None
        block_indices = list(range(depth - blocks_to_swap, depth))
        return cls(block_indices, offload_device)

    def activate(
        self, blocks: Iterable[nn.Module], compute_device: torch.device, grad_enabled: bool
    ) -> bool:
        if compute_device == self.offload_device:
            return False
        blocks_list = list(blocks)
        self._ensure_backward_hooks(blocks_list, compute_device, grad_enabled)
        self.mark_blocks_for_offload(blocks_list)
        return True

    def is_managed_block(self, index: int) -> bool:
        return index in self.block_indices

    def stream_in(self, block: nn.Module, device: torch.device):
        self._move_module(block, device)

    def stream_out(self, block: nn.Module):
        self._move_module(block, self.offload_device)

    def mark_blocks_for_offload(self, blocks: List[nn.Module]):
        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            self._move_module(blocks[idx], self.offload_device)

    def _clear_backward_hooks(self):
        for handle in self._backward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._backward_hooks.clear()
        self._backward_hook_device = None

    def _ensure_backward_hooks(
        self, blocks: List[nn.Module], compute_device: torch.device, grad_enabled: bool
    ) -> None:
        if not grad_enabled:
            return
        if self._backward_hook_device == compute_device and self._backward_hooks:
            return
        self._clear_backward_hooks()

        for idx, block in enumerate(blocks):
            if not self.is_managed_block(idx):
                continue

            def _make_pre_hook():
                def _pre_hook(module, _grad_output):
                    self.stream_in(module, compute_device)
                    return None

                return _pre_hook

            def _make_post_hook():
                def _post_hook(module, _grad_input, _grad_output):
                    self.stream_out(module)
                    return None

                return _post_hook

            self._backward_hooks.append(block.register_full_backward_pre_hook(_make_pre_hook()))
            self._backward_hooks.append(block.register_full_backward_hook(_make_post_hook()))

        self._backward_hook_device = compute_device

    def _move_module(self, module: nn.Module, device: torch.device):
        if _module_on_device(module, device):
            return
        with torch.no_grad():
            module.to(device)


class LTX2ModelOffloader(ModelOffloader):
    """LTX-2 local offloader that avoids GPU preloading for swap blocks."""

    def __init__(self, *args, swap_norms: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.swap_norms = swap_norms

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]) -> None:
        global _LOGGED_SWAP_BYTES, _LOGGED_FIRST_PARAM
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward (LTX2)")
        diag_enabled = os.getenv("LTX2_SWAP_DIAG", "0") == "1"
        if diag_enabled:
            keep_idx = 0
            swap_idx = max(0, self.num_blocks - self.blocks_to_swap)
            if blocks:
                _summarize_block_tensors(blocks[keep_idx], "before_keep_block")
                if swap_idx < len(blocks):
                    _summarize_block_tensors(blocks[swap_idx], "before_swap_block")
        _log_cuda_memory("before_prepare_blocks")

        use_pinned = self.use_pinned_memory and os.getenv("LTX2_SWAP_PINNED", "1") == "1"
        split_idx = max(0, self.num_blocks - self.blocks_to_swap)
        for block in blocks[0 : split_idx]:
            block.to(self.device)
            weighs_to_device(block, self.device, use_pinned=use_pinned)

        cpu_device = torch.device("cpu")
        for block in blocks[split_idx :]:
            if self.swap_norms:
                # Move whole block to GPU, then swap Linear AND norm weights to CPU
                block.to(self.device)
                if _SKIP_AUDIO_SWAP or _SKIP_CROSS_ATTN_SWAP:
                    _move_block_params_excluding_audio(
                        block,
                        cpu_device,
                        include_norms=True,
                        use_pinned=use_pinned,
                    )
                else:
                    params_to_device(block, cpu_device, include_norms=True, use_pinned=use_pinned)
            else:
                # Move only non-linear params/buffers to GPU to avoid peak VRAM spikes.
                _move_non_linear_params(block, self.device)
                if _SKIP_AUDIO_SWAP or _SKIP_CROSS_ATTN_SWAP:
                    _move_block_params_excluding_audio(
                        block,
                        cpu_device,
                        include_norms=False,
                        use_pinned=use_pinned,
                    )
                else:
                    weighs_to_device(block, cpu_device, use_pinned=use_pinned)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)
        _log_cuda_memory("after_prepare_blocks")
        if not _LOGGED_SWAP_BYTES:
            split_idx = max(0, self.num_blocks - self.blocks_to_swap)
            _log_block_cuda_bytes(blocks, split_idx)
            _LOGGED_SWAP_BYTES = True
        if not _LOGGED_FIRST_PARAM:
            _log_first_param(blocks)
            _LOGGED_FIRST_PARAM = True
        if diag_enabled:
            keep_idx = 0
            swap_idx = max(0, self.num_blocks - self.blocks_to_swap)
            if blocks:
                _summarize_block_tensors(blocks[keep_idx], "after_keep_block")
                if swap_idx < len(blocks):
                    _summarize_block_tensors(blocks[swap_idx], "after_swap_block")

