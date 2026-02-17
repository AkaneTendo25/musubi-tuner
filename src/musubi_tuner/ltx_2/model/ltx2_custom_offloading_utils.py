from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import os
import time
from typing import Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Keep these functions here for portability, and private to avoid confusion with the ones in device_utils.py
def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


# Cache for pinned buffers to avoid reallocation overhead: {param_id: pinned_tensor}
_pinned_buffer_cache = {}
_FIRST_SWAP_TRACED = False  # Flag for first-swap VRAM tracing
_LOGGED_FULL_BLOCK_PINNED = False

def _trace_first_swap_vram(tag: str):
    """Trace VRAM during first swap operation."""
    global _FIRST_SWAP_TRACED
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        m = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[VRAM_FIRST_SWAP] {tag}: alloc={a:.2f}GB res={r:.2f}GB max={m:.2f}GB")
_LOGGED_FP8_UPCAST = False
_ALLOW_FP8_OFFLOAD_UPCAST = True
_FP8_OFFLOAD_RESTORE_BF16 = True
_FP8_OFFLOAD_RESTORE_STOCHASTIC = False
_FP8_OFFLOAD_KEEP_FP8 = False
_FP8_ORIG_DTYPE = {}
_FP8_BUFFER_ORIG_DTYPE = {}
_FP8_CPU_CACHE = {}
_FP8_DTYPES = tuple(
    dt for dt in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)) if dt is not None
)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def set_fp8_offload_upcast(enable: bool) -> None:
    """Allow FP8 weights to be offloaded to CPU by upcasting to bf16."""
    global _ALLOW_FP8_OFFLOAD_UPCAST
    _ALLOW_FP8_OFFLOAD_UPCAST = bool(enable)


def set_fp8_offload_restore_bf16(enable: bool) -> None:
    """Keep FP8-offloaded weights in bf16 on GPU after restore (avoid FP8 round-trip)."""
    global _FP8_OFFLOAD_RESTORE_BF16
    _FP8_OFFLOAD_RESTORE_BF16 = bool(enable)


def set_fp8_offload_restore_stochastic(enable: bool) -> None:
    """Apply experimental stochastic rounding noise before restoring FP8 weights."""
    global _FP8_OFFLOAD_RESTORE_STOCHASTIC
    _FP8_OFFLOAD_RESTORE_STOCHASTIC = bool(enable)


def set_fp8_offload_keep_fp8(enable: bool) -> None:
    """Keep FP8 weights in FP8 on CPU to avoid bf16 round-trips."""
    global _FP8_OFFLOAD_KEEP_FP8
    _FP8_OFFLOAD_KEEP_FP8 = bool(enable)

def weighs_to_device(layer: nn.Module, device: torch.device, skip_trainable: bool = True, use_pinned: bool = False):
    """Move layer weights to device.
    
    Args:
        layer: Module to process
        device: Target device
        skip_trainable: If True AND target is CPU, skip parameters with requires_grad=True
        use_pinned: If True and target is CPU, use cached pinned memory.
    """
    non_blocking = device.type != "cpu"
    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and device.type == "cpu"
    
    for module in layer.modules():
        if module.__class__.__name__.endswith("Linear"):
            for attr in ["weight", "bias", "scale_weight"]:
                p = getattr(module, attr, None)
                if p is not None and isinstance(p, (torch.Tensor, torch.nn.Parameter)):
                    if device.type != "cpu" and id(p) in _FP8_ORIG_DTYPE:
                        # Restore FP8-offloaded weights to GPU.
                        # If restore_bf16 is enabled, keep bf16 and drop FP8 tracking to avoid extra overhead.
                        if _FP8_OFFLOAD_RESTORE_BF16:
                            p.data = p.data.to(device, non_blocking=non_blocking)
                            _FP8_ORIG_DTYPE.pop(id(p), None)
                            continue
                        target_dtype = _FP8_ORIG_DTYPE[id(p)]
                        if (
                            _FP8_OFFLOAD_RESTORE_STOCHASTIC
                            and target_dtype in _FP8_DTYPES
                            and torch.is_floating_point(p.data)
                        ):
                            # Experimental: add small relative noise before FP8 cast to reduce bias.
                            t = p.data.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                            noise = torch.empty_like(t).uniform_(-0.5, 0.5)
                            t = t + noise * (t.abs() + 1.0) * 1e-3
                            p.data = t.to(dtype=target_dtype)
                        else:
                            p.data = p.data.to(device, dtype=target_dtype, non_blocking=non_blocking)
                        continue
                    if device.type == "cpu" and _is_fp8_dtype(p.dtype):
                        if _FP8_OFFLOAD_KEEP_FP8:
                            try:
                                p.data = p.data.to(device, non_blocking=non_blocking)
                            except Exception:
                                p.data = p.data.to(device)
                            continue
                        if not _ALLOW_FP8_OFFLOAD_UPCAST:
                            continue
                        # Explicit opt-in: upcast FP8 to bf16 on CPU, then restore FP8 on GPU.
                        global _LOGGED_FP8_UPCAST
                        if not _LOGGED_FP8_UPCAST:
                            logging.getLogger(__name__).warning(
                                "FP8 offload upcast enabled: FP8 weights are cast to bf16 on CPU."
                            )
                            _LOGGED_FP8_UPCAST = True
                        if not _FP8_OFFLOAD_RESTORE_BF16 or _FP8_OFFLOAD_RESTORE_STOCHASTIC:
                            _FP8_ORIG_DTYPE[id(p)] = p.dtype
                        # Cache fp8 weights on CPU as bf16 (pin if requested) to avoid CPU float8 tensors.
                        try:
                            if use_pinned:
                                buf = _FP8_CPU_CACHE.get(id(p))
                                if buf is None or buf.shape != p.data.shape or buf.dtype != torch.bfloat16:
                                    buf = torch.empty_like(p.data, device="cpu", dtype=torch.bfloat16, pin_memory=True)
                                    _FP8_CPU_CACHE[id(p)] = buf
                                buf.copy_(p.data.to(dtype=torch.bfloat16), non_blocking=non_blocking)
                                p.data = buf
                            else:
                                p.data = p.data.to(device, dtype=torch.bfloat16, non_blocking=non_blocking)
                        except Exception:
                            # Fallback to fp16 if bf16 conversion fails
                            if use_pinned:
                                buf = _FP8_CPU_CACHE.get(id(p))
                                if buf is None or buf.shape != p.data.shape or buf.dtype != torch.float16:
                                    buf = torch.empty_like(p.data, device="cpu", dtype=torch.float16, pin_memory=True)
                                    _FP8_CPU_CACHE[id(p)] = buf
                                buf.copy_(p.data.to(dtype=torch.float16), non_blocking=non_blocking)
                                p.data = buf
                            else:
                                p.data = p.data.to(device, dtype=torch.float16, non_blocking=non_blocking)
                        continue
                    # Skip trainable parameters only when offloading to CPU
                    if should_skip_trainable and hasattr(p, 'requires_grad') and p.requires_grad:
                        continue
                    
                    if use_pinned and device.type == "cpu":
                        # Reuse or allocate pinned buffer
                        pid = id(p)
                        if (
                            pid not in _pinned_buffer_cache
                            or _pinned_buffer_cache[pid].shape != p.data.shape
                            or _pinned_buffer_cache[pid].dtype != p.data.dtype
                        ):
                            # Allocate new pinned buffer
                            _pinned_buffer_cache[pid] = torch.empty_like(p, device="cpu", pin_memory=True)
                        
                        target_buffer = _pinned_buffer_cache[pid]
                        target_buffer.copy_(p.data, non_blocking=non_blocking)
                        p.data = target_buffer
                    else:
                        p.data = p.data.to(device, non_blocking=non_blocking)
            
            # LoRA Handling for monkey-patched modules
            if should_skip_trainable:
                continue
            forward_self = getattr(getattr(module, "forward", None), "__self__", None)
            if forward_self is not None and forward_self is not module:
                for lora_name in ["lora_down", "lora_up"]:
                    lora_mod = getattr(forward_self, lora_name, None)
                    if isinstance(lora_mod, torch.nn.Module):
                        lora_mod.to(device)



def params_to_device(layer: nn.Module, device: torch.device, include_norms: bool = False, use_pinned: bool = False, skip_trainable: bool = True):
    """Move module parameters to device, optionally including normalization layers.

    Args:
        layer: Module to process
        device: Target device
        include_norms: If True, also move RMSNorm/LayerNorm weights (more VRAM savings, more overhead)
        use_pinned: If True and target is CPU, pin memory. If target is GPU, assumes source is pinned for async transfer.
        skip_trainable: If True AND target is CPU, skip parameters with requires_grad=True (protects LoRA params)
    """
    non_blocking = device.type != "cpu"
    # Only skip trainable parameters when moving TO CPU (offloading), not when loading TO GPU
    should_skip_trainable = skip_trainable and device.type == "cpu"

    # Patterns for normalization layers
    norm_patterns = ("RMSNorm", "LayerNorm", "GroupNorm", "BatchNorm")

    for module in layer.modules():
        class_name = module.__class__.__name__

        # Linear layers
        if class_name.endswith("Linear"):
            for attr in ["weight", "bias", "scale_weight"]:
                p = getattr(module, attr, None)
                if p is not None and isinstance(p, (torch.Tensor, torch.nn.Parameter)):
                    # Skip trainable (LoRA) parameters when offloading to CPU
                    if should_skip_trainable and hasattr(p, 'requires_grad') and p.requires_grad:
                        continue
                    if device.type == "cpu" and use_pinned:
                        # Reuse or allocate pinned buffer
                        pid = id(p)
                        if (
                            pid not in _pinned_buffer_cache
                            or _pinned_buffer_cache[pid].shape != p.data.shape
                            or _pinned_buffer_cache[pid].dtype != p.data.dtype
                        ):
                            _pinned_buffer_cache[pid] = torch.empty_like(p, device="cpu", pin_memory=True)
                        target_buffer = _pinned_buffer_cache[pid]
                        target_buffer.copy_(p.data, non_blocking=non_blocking)
                        p.data = target_buffer
                    else:
                        p.data = p.data.to(device, non_blocking=non_blocking)
        
        # Normalization layers (if enabled)
        elif include_norms and class_name.endswith(norm_patterns):
            for attr in ["weight", "bias"]:
                p = getattr(module, attr, None)
                if p is not None and isinstance(p, (torch.Tensor, torch.nn.Parameter)):
                    # Skip trainable (LoRA) parameters when offloading to CPU
                    if should_skip_trainable and hasattr(p, 'requires_grad') and p.requires_grad:
                        continue
                    if device.type == "cpu" and use_pinned:
                        # Reuse or allocate pinned buffer
                        pid = id(p)
                        if pid not in _pinned_buffer_cache:
                            _pinned_buffer_cache[pid] = torch.empty_like(p, device="cpu", pin_memory=True)
                        target_buffer = _pinned_buffer_cache[pid]
                        target_buffer.copy_(p.data, non_blocking=non_blocking)
                        p.data = target_buffer
                    else:
                        p.data = p.data.to(device, non_blocking=non_blocking)


class Offloader:
    """
    common offloading class
    """

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = min(blocks_to_swap, num_blocks)
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        # check if debug is enabled from os environment variable
        if not debug:
            import os

            debug = os.getenv("LTX2_OFFLOADER_DEBUG", "0") == "1"

        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for cuda offloading without large pinned memory. These are pinned memory buffers to speed up the transfer between CPU and GPU
        # We create one staging buffer per transfer direction (A: GPU to CPU, B: CPU to GPU)
        self.staging_buffer_a = None
        self.staging_buffer_b = None

        # Pinned buffer for cuda offloading with pinned memory. We need only one pinned buffer per layer transfer
        self.pinned_buffer = None

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        global _FIRST_SWAP_TRACED
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        debug_print = False
        if self.debug:
            debug_print = self.debug_block_count % 10 == 0
            self.debug_block_count += 1

        class Timer:
            def __init__(self, enabled=False):
                self.enabled = enabled
                self.totals = defaultdict(float)
                self.start_time = time.perf_counter()

            @contextmanager
            def section(self, name):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = Timer(enabled=debug_print)

        weight_swap_jobs = []

        # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules. kept here for reference:
        # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        #     print(module_to_cpu.__class__, module_to_cuda.__class__)
        #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
        #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

        with T.section("find modules"):
            modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()} if layer_to_cpu is not None else {}
            for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
                # We only offload Linear layers for now to avoid excessive overhead with small parameters
                if not module_to_cuda.__class__.__name__.endswith("Linear"):
                    continue

                module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                
                for attr_name in ["weight", "bias", "scale_weight"]:
                    cuda_param = getattr(module_to_cuda, attr_name, None)
                    if cuda_param is None or not isinstance(cuda_param, (torch.nn.Parameter, torch.Tensor)):
                        continue
                    if module_to_cpu is not None:
                        cpu_param = getattr(module_to_cpu, attr_name, None)
                        if (
                            cpu_param is not None 
                            and isinstance(cpu_param, (torch.nn.Parameter, torch.Tensor))
                            and cpu_param.shape == cuda_param.shape
                            and cpu_param.dtype == cuda_param.dtype
                        ):
                            # Swap jobs: (parent_to_cpu, parent_to_cuda, cuda_data, cpu_data, attr_name)
                            # cpu_param.data is currently on GPU, cuda_param.data is currently on CPU
                            weight_swap_jobs.append(
                                (module_to_cpu, module_to_cuda, cpu_param.data, cuda_param.data, attr_name)
                            )
                            continue

                    # Fallback: if no counterpart for this module/attribute, ensure it's on the target device
                    if cuda_param.device.type != device.type:
                        cuda_param.data = cuda_param.data.to(device)

                # LoRA Handling for monkey-patched modules
                if module_to_cpu is not None:
                    cuda_forward_self = getattr(getattr(module_to_cuda, "forward", None), "__self__", None)
                    cpu_forward_self = getattr(getattr(module_to_cpu, "forward", None), "__self__", None)

                    if cuda_forward_self is not None and cuda_forward_self is not module_to_cuda and cpu_forward_self is not None:
                        for lora_name in ["lora_down", "lora_up"]:
                            cuda_lora = getattr(cuda_forward_self, lora_name, None)
                            cpu_lora = getattr(cpu_forward_self, lora_name, None)

                            if isinstance(cuda_lora, torch.nn.Module) and isinstance(cpu_lora, torch.nn.Module):
                                for attr_name_lora in ["weight", "bias"]:
                                    cuda_p = getattr(cuda_lora, attr_name_lora, None)
                                    cpu_p = getattr(cpu_lora, attr_name_lora, None)

                                    if hasattr(cuda_p, "data") and hasattr(cpu_p, "data"):
                                        weight_swap_jobs.append(
                                            (cpu_lora, cuda_lora, cpu_p.data, cuda_p.data, attr_name_lora)
                                        )

        with T.section("synchronize before swap"):
            torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value by ensuring offloading layer's calculation is done

        if not self.use_pinned_memory:
            # Minimize using pinned memory for lower shared GPU RAM usage
            stream = self.stream
            with torch.cuda.stream(stream):
                def _staging_buffer_valid() -> bool:
                    if self.staging_buffer_a is None or self.staging_buffer_b is None:
                        return False
                    if len(self.staging_buffer_a) != len(weight_swap_jobs):
                        return False
                    if len(self.staging_buffer_b) != len(weight_swap_jobs):
                        return False
                    for sbuf_a, sbuf_b, (_, _, cuda_data_view, _, _) in zip(
                        self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                    ):
                        if sbuf_a.shape != cuda_data_view.shape or sbuf_a.dtype != cuda_data_view.dtype:
                            return False
                        if sbuf_b.shape != cuda_data_view.shape or sbuf_b.dtype != cuda_data_view.dtype:
                            return False
                    return True

                if not _staging_buffer_valid():
                    if not _FIRST_SWAP_TRACED:
                        _trace_first_swap_vram("BEFORE staging buffer creation")
                    # Create staging buffer as pinned memory (as shared GPU ram). We specify device for correct pinning on multi-GPU systems
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu")
                        for _, _, cuda_data_view, _, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu")
                        for _, _, cuda_data_view, _, _ in weight_swap_jobs
                    ]
                    if not _FIRST_SWAP_TRACED:
                        _trace_first_swap_vram("AFTER staging buffer creation (CPU buffers)")
                        _FIRST_SWAP_TRACED = True

                # Copy weights to staging buffers and record events
                event_b = None
                for sbuf_a, sbuf_b, (parent_to_cpu, parent_to_cuda, cuda_data_view, cpu_data_view, attr_name) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    # CUDA to staging buffer A, non-blocking copy
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                        event_a.record(stream)

                    # Wait for staging buffer B to be ready
                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()  # synchronize is needed to wait CPU process. wait_event does not work here because it waits on GPU side only

                    # CPU to staging buffer B, CPU to pinned CPU, synchronous copy. Can overlap with CUDA to staging buffer A
                    with T.section("cpu to staging B"):
                        # Making this multithreaded does not help, and 'non_blocking=True' does not help either.
                        sbuf_b.copy_(cpu_data_view)  # BOTTLENECK

                    # Wait for staging buffer A to be ready, and CUDA data view can be reused
                    with T.section("wait staging A"):
                        event_a.synchronize()

                    # Staging buffer B to CUDA, non-blocking copy.
                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_data_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    # Staging buffer A to CPU, synchronous copy. Can overlap with staging buffer B to CUDA
                    with T.section("staging A to CPU"):
                        cpu_data_view.copy_(sbuf_a)  # BOTTLENECK

            for sbuf_a, sbuf_b, (parent_to_cpu, parent_to_cuda, cuda_data_view, cpu_data_view, attr_name) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                # Update references
                getattr(parent_to_cuda, attr_name).data = cuda_data_view
                getattr(parent_to_cpu, attr_name).data = cpu_data_view

            sync_event = event_b  # final sync event for CPU to CUDA copy

        else:
            # Use pinned memory for faster transfer between CPU and GPU, but it requires more memory
            def _pinned_buffer_valid() -> bool:
                if self.pinned_buffer is None:
                    return False
                if len(self.pinned_buffer) != len(weight_swap_jobs):
                    return False
                for buf, (_, _, cuda_data_view, _, _) in zip(self.pinned_buffer, weight_swap_jobs):
                    if buf.shape != cuda_data_view.shape or buf.dtype != cuda_data_view.dtype:
                        return False
                return True

            if not _pinned_buffer_valid():
                if not _FIRST_SWAP_TRACED:
                    _trace_first_swap_vram("BEFORE pinned buffer creation")
                with torch.cuda.stream(self.stream):
                    # Create pinned buffer as pinned memory (as shared GPU ram). We specify device for correct pinning on multi-GPU systems
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
                if not _FIRST_SWAP_TRACED:
                    _trace_first_swap_vram("AFTER pinned buffer creation")
                    _FIRST_SWAP_TRACED = True
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]  # Waiting events for GPU to CPU non-blocking copy

            def _copy_weights_to_cpu():
                for event, module_pin_buf, (parent_to_cpu, parent_to_cuda, cuda_data_view, cpu_data_view, attr_name) in zip(
                    events, self.pinned_buffer, weight_swap_jobs
                ):
                    # CUDA to CPU, non-blocking copy
                    with torch.cuda.stream(self.stream):
                        with T.section("cuda to cpu"):
                            module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                            event.record(self.stream)

            # Copy weights to CPU (retry once if pinned buffer shape mismatch slips through)
            try:
                _copy_weights_to_cpu()
            except RuntimeError as e:
                if "must match the size of tensor" in str(e):
                    with torch.cuda.stream(self.stream):
                        self.pinned_buffer = [
                            torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                            for _, _, cuda_data_view, _, _ in weight_swap_jobs
                        ]
                    self.stream.synchronize()
                    _copy_weights_to_cpu()
                else:
                    raise

            # CPU to CUDA
            for event, (parent_to_cpu, parent_to_cuda, cuda_data_view, cpu_data_view, attr_name) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    # Wait for cuda_data_view to be ready
                    with T.section("wait cpu"):
                        self.stream.wait_event(event)

                    # CPU to CUDA, non-blocking copy
                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            # Update references
            for module_pin_buf, (parent_to_cpu, parent_to_cuda, cuda_data_view, cpu_data_view, attr_name) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                getattr(parent_to_cuda, attr_name).data = cuda_data_view
                getattr(parent_to_cpu, attr_name).data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)  # CPU data view can be reused as pinned buffer

            # Reuse released pinned buffers
            if not released_pinned_buffer[0].is_pinned():
                # In first time, we need to create pinned buffers because offloaded weights are not pinned yet
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        if debug_print:
            print(f"[{self.block_type}] Weight swap timing at {self.debug_block_count - 1}:")
            for name, total in T.totals.items():
                print(f"  {name}: {total * 1000:.2f}ms")
            print(
                f"Overall time: {(time.perf_counter() - T.start_time) * 1000:.2f}ms, total time in sections: {sum(T.totals.values()) * 1000:.2f}ms"
            )
        # print(
        #     f"[{self.block_type}] Swapped weights in {time.perf_counter() - start_time:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
        # )

        return sync_event

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            sync_event = self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            sync_event = None
        return sync_event

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _move_module_with_optional_pinning(self, module: nn.Module, target_device: torch.device) -> None:
        """Move a full module, using pinned CPU buffers when enabled.

        This is used by aggressive full-block swap mode where entire blocks are streamed
        CPU<->GPU. If pinned memory is disabled, behavior is identical to module.to(...).
        """
        target_device = torch.device(target_device)

        # Keep legacy behavior unless we're on CUDA and pinned memory is explicitly enabled.
        if not (self.cuda_available and self.use_pinned_memory):
            module.to(target_device)
            return

        # Pinned path currently targets CUDA<->CPU block streaming only.
        if target_device.type not in {"cpu", "cuda"}:
            module.to(target_device)
            return

        global _LOGGED_FULL_BLOCK_PINNED
        if not _LOGGED_FULL_BLOCK_PINNED:
            logger.info("LTX-2 swap: full-block pinned transfer path enabled")
            _LOGGED_FULL_BLOCK_PINNED = True

        non_blocking = target_device.type != "cpu"

        def _to_cpu_with_cache(src: torch.Tensor, cache_key, *, force_dtype: torch.dtype = None) -> torch.Tensor:
            cpu_dtype = force_dtype if force_dtype is not None else src.dtype
            cached = _pinned_buffer_cache.get(cache_key)
            if (
                cached is None
                or cached.shape != src.shape
                or cached.dtype != cpu_dtype
            ):
                cached = torch.empty_like(src, device="cpu", dtype=cpu_dtype, pin_memory=True)
                _pinned_buffer_cache[cache_key] = cached
            if force_dtype is not None:
                cached.copy_(src.to(dtype=cpu_dtype), non_blocking=True)
            else:
                cached.copy_(src, non_blocking=True)
            return cached

        def _restore_fp8_tensor(src: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
            if (
                _FP8_OFFLOAD_RESTORE_STOCHASTIC
                and target_dtype in _FP8_DTYPES
                and torch.is_floating_point(src)
            ):
                t = src.to(device=target_device, dtype=torch.float32, non_blocking=non_blocking)
                noise = torch.empty_like(t).uniform_(-0.5, 0.5)
                t = t + noise * (t.abs() + 1.0) * 1e-3
                return t.to(dtype=target_dtype)
            return src.to(target_device, dtype=target_dtype, non_blocking=non_blocking)

        for submodule in module.modules():
            # Parameters
            for p in submodule.parameters(recurse=False):
                if p is None:
                    continue
                if p.data.device == target_device:
                    continue

                if target_device.type == "cpu":
                    key = ("param", id(p))
                    src = p.data

                    if _is_fp8_dtype(src.dtype):
                        moved = False

                        # Prefer true FP8 transfer (parity with non-LTX offloader behavior).
                        try:
                            if self.use_pinned_memory:
                                p.data = _to_cpu_with_cache(src, key)
                            else:
                                p.data = src.to("cpu", non_blocking=True)
                            _FP8_ORIG_DTYPE.pop(id(p), None)
                            moved = True
                        except Exception:
                            moved = False

                        # Fallback for platforms/torch builds that cannot keep FP8 on CPU pinned buffers.
                        if not moved:
                            _FP8_ORIG_DTYPE[id(p)] = src.dtype
                            try:
                                if self.use_pinned_memory:
                                    p.data = _to_cpu_with_cache(src, key, force_dtype=torch.bfloat16)
                                else:
                                    p.data = src.to("cpu", dtype=torch.bfloat16, non_blocking=True)
                            except Exception:
                                if self.use_pinned_memory:
                                    p.data = _to_cpu_with_cache(src, key, force_dtype=torch.float16)
                                else:
                                    p.data = src.to("cpu", dtype=torch.float16, non_blocking=True)
                    else:
                        if self.use_pinned_memory:
                            p.data = _to_cpu_with_cache(src, key)
                        else:
                            p.data = src.to("cpu", non_blocking=True)
                else:
                    src = p.data
                    fp8_dtype = _FP8_ORIG_DTYPE.pop(id(p), None)
                    if fp8_dtype is None:
                        p.data = src.to(target_device, non_blocking=non_blocking)
                    elif _FP8_OFFLOAD_RESTORE_BF16:
                        p.data = src.to(target_device, non_blocking=non_blocking)
                    else:
                        p.data = _restore_fp8_tensor(src, fp8_dtype)

            # Buffers
            for buf_name, buf in submodule.named_buffers(recurse=False):
                if buf is None or not isinstance(buf, torch.Tensor):
                    continue
                if buf.device == target_device:
                    continue

                key = ("buffer", id(submodule), buf_name)

                if target_device.type == "cpu":
                    src = buf

                    if _is_fp8_dtype(src.dtype):
                        moved = False
                        try:
                            if self.use_pinned_memory:
                                submodule._buffers[buf_name] = _to_cpu_with_cache(src, key)
                            else:
                                submodule._buffers[buf_name] = src.to("cpu", non_blocking=True)
                            _FP8_BUFFER_ORIG_DTYPE.pop(key, None)
                            moved = True
                        except Exception:
                            moved = False

                        if not moved:
                            _FP8_BUFFER_ORIG_DTYPE[key] = src.dtype
                            try:
                                if self.use_pinned_memory:
                                    submodule._buffers[buf_name] = _to_cpu_with_cache(src, key, force_dtype=torch.bfloat16)
                                else:
                                    submodule._buffers[buf_name] = src.to("cpu", dtype=torch.bfloat16, non_blocking=True)
                            except Exception:
                                if self.use_pinned_memory:
                                    submodule._buffers[buf_name] = _to_cpu_with_cache(src, key, force_dtype=torch.float16)
                                else:
                                    submodule._buffers[buf_name] = src.to("cpu", dtype=torch.float16, non_blocking=True)
                    else:
                        if self.use_pinned_memory:
                            submodule._buffers[buf_name] = _to_cpu_with_cache(src, key)
                        else:
                            submodule._buffers[buf_name] = src.to("cpu", non_blocking=True)
                else:
                    src = buf
                    fp8_dtype = _FP8_BUFFER_ORIG_DTYPE.pop(key, None)
                    if fp8_dtype is None:
                        submodule._buffers[buf_name] = src.to(target_device, non_blocking=non_blocking)
                    elif _FP8_OFFLOAD_RESTORE_BF16:
                        submodule._buffers[buf_name] = src.to(target_device, non_blocking=non_blocking)
                    else:
                        submodule._buffers[buf_name] = _restore_fp8_tensor(src, fp8_dtype)

    def _submit_load_block(self, blocks, block_idx):
        """Single-direction load: move ENTIRE block from CPU to GPU without swapping another out."""
        def load_block(bidx, block):
            if self.debug:
                start_time = time.perf_counter()
                print(f"[{self.block_type}] Load block {bidx} to {'CUDA' if self.cuda_available else 'device'}")

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            # Move ENTIRE block to GPU (all params and buffers, not just Linear weights)
            self._move_module_with_optional_pinning(block, self.device)

            if self.cuda_available:
                torch.cuda.current_stream().synchronize()
                sync_event = torch.cuda.current_stream().record_event()
            else:
                sync_event = None

            if self.debug:
                print(f"[{self.block_type}] Loaded block {bidx} in {time.perf_counter() - start_time:.2f}s")
            return bidx, bidx, sync_event

        block = blocks[block_idx]
        self.futures[block_idx] = self.thread_pool.submit(load_block, block_idx, block)

    def _submit_unload_block(self, blocks, block_idx, sync: bool = True):
        """Single-direction unload: move ENTIRE block from GPU to CPU.

        Args:
            blocks: List of blocks
            block_idx: Index of block to unload
            sync: If True, wait for unload to complete (default True to prevent VRAM accumulation)
        """
        def unload_block(bidx, block):
            if self.debug:
                start_time = time.perf_counter()
                print(f"[{self.block_type}] Unload block {bidx} to CPU")

            # Move ENTIRE block to CPU
            self._move_module_with_optional_pinning(block, torch.device("cpu"))

            # Synchronize to ensure transfer is complete
            if self.cuda_available:
                torch.cuda.synchronize()

            if self.debug:
                print(f"[{self.block_type}] Unloaded block {bidx} in {time.perf_counter() - start_time:.2f}s")
            return bidx, bidx, None

        block = blocks[block_idx]
        if sync:
            # Synchronous unload to prevent VRAM accumulation
            unload_block(block_idx, block)
        else:
            # Async unload (original behavior)
            self.thread_pool.submit(unload_block, block_idx, block)

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available and sync_event is not None:
            # this does not wait CPU side, so the log below should be immediate when pinned memory is used
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        # switching must wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)

        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        split_idx = max(0, self.num_blocks - self.blocks_to_swap)
        for b in blocks[0 : split_idx]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        cpu_device = torch.device("cpu")
        for b in blocks[split_idx :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, cpu_device)  # make sure weights are on cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        aggressive_train_swap = os.getenv("LTX2_SWAP_TRAIN_FULL", "0") == "1"
        split_idx = max(0, self.num_blocks - self.blocks_to_swap)

        if not self.forward_only and not aggressive_train_swap:
            # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        # Training mode with aggressive swap: Stream ENTIRE blocks through GPU
        # - Blocks 0 to split_idx-1: Stay on GPU permanently (for fast backward)
        # - Blocks split_idx to N-1: Stream through (load → compute → unload)
        # Uses full block moves (block.to(device)) instead of Linear-only swaps
        if not self.forward_only and aggressive_train_swap:
            # Early blocks stay on GPU - do not swap them
            if block_idx < split_idx:
                # Preload the first swapped block when we're about to reach it
                if block_idx == split_idx - 1 and split_idx < self.num_blocks:
                    self._submit_load_block(blocks, split_idx)
                return

            # Swap range: after executing block N, unload it and load N+1
            # Use full block moves instead of Linear-only swaps for complete VRAM savings
            self._submit_unload_block(blocks, block_idx)

            # Load next block to GPU
            if block_idx + 1 < self.num_blocks:
                self._submit_load_block(blocks, block_idx + 1)
            else:
                # Last block - load first swapped block for backward pass
                self._submit_load_block(blocks, split_idx)
            return

        # Forward-only (inference) mode: Use traditional swap strategies
        # We use two strategies here for forward-only offloading:
        # 1. If blocks_to_swap is less than half of num_blocks, we swap the num_blocks blocks without wrapping around.
        #   This reduces the number of swaps, so it is especially useful for small blocks_to_swap or lightweight models like Qwen-Image
        # 2. If blocks_to_swap is more than half of num_blocks, we swap the blocks with wrapping around.
        #   This is the common strategy used in most offloading implementations. It transfers all blocks in a wrapping manner.
        #   This is useful for large blocks_to_swap or heavyweight models like Wan/HunyuanVideo, where the transfer time is less significant compared to computation time.
        #
        # LTX2_SWAP_NO_WRAPAROUND=1 forces strategy 1 (no wrap-around) regardless of blocks_to_swap count.
        # This can reduce VRAM spikes when swapping many blocks.

        # current block to swap out (to CPU)
        block_idx_to_cpu = block_idx

        no_wraparound = os.getenv("LTX2_SWAP_NO_WRAPAROUND", "1") == "1"
        use_strategy_1 = self.blocks_to_swap < (self.num_blocks // 2) or no_wraparound

        if use_strategy_1:
            # strategy 1: no wrap around
            # If the current block is in the middle blocks that are not swapped, do nothing
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                # move the next block to cuda
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                # move the previous block to cuda
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            # strategy 2: with wrap around
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
