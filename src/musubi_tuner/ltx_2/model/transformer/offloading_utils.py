import logging
import os
import torch
import torch.nn as nn

from typing import Iterable, List, Optional

from musubi_tuner.modules.custom_offloading_utils import (
    ModelOffloader,
    _clean_memory_on_device,
    _synchronize_device,
    weighs_to_device,
)

logger = logging.getLogger(__name__)


def _log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info("LTX-2 swap mem [%s]: cuda_allocated=%.2fGB cuda_reserved=%.2fGB", tag, allocated, reserved)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


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
    """Stream full blocks between devices (SimpleTuner-style) for LTX-2."""

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

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]) -> None:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward (LTX2)")
        diag_enabled = os.getenv("MUSUBI_TUNER_LTX2_SWAP_DIAG", "0") == "1"
        if diag_enabled:
            keep_idx = 0
            swap_idx = max(0, self.num_blocks - self.blocks_to_swap)
            if blocks:
                _summarize_block_tensors(blocks[keep_idx], "before_keep_block")
                if swap_idx < len(blocks):
                    _summarize_block_tensors(blocks[swap_idx], "before_swap_block")
        _log_cuda_memory("before_prepare_blocks")

        for block in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            block.to(self.device)
            weighs_to_device(block, self.device)

        cpu_device = torch.device("cpu")
        for block in blocks[self.num_blocks - self.blocks_to_swap :]:
            block.to(cpu_device)
            weighs_to_device(block, cpu_device)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)
        _log_cuda_memory("after_prepare_blocks")
        if diag_enabled:
            keep_idx = 0
            swap_idx = max(0, self.num_blocks - self.blocks_to_swap)
            if blocks:
                _summarize_block_tensors(blocks[keep_idx], "after_keep_block")
                if swap_idx < len(blocks):
                    _summarize_block_tensors(blocks[swap_idx], "after_swap_block")
