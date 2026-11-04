"""Dense H3 checkpointing for logical tensors backed by packed ring storage.

Accelerate/safetensors' shared-storage deduplication is inappropriate for the
disjoint views in the trainable ring. Keep this path local to the dense trainer.
"""

import json
import logging
import os
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

from musubi_tuner.utils.safetensors_utils import mem_eff_save_file


MANIFEST = "h3_dense_state.json"
INCOMPLETE = "h3_dense_state.incomplete"
FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


def canonical_name(name):
    return ".".join(part for part in name.split(".") if part != "_orig_mod")


def logical_state(transformer):
    result = {}
    for name, tensor in transformer.state_dict().items():
        name = canonical_name(name)
        if name in result:
            raise ValueError(f"dense state canonical-name collision: {name}")
        result[name] = tensor
    return result


def optimizer_signature(transformer, optimizer):
    names = {id(parameter): canonical_name(name) for name, parameter in transformer.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        entries = []
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise ValueError("dense optimizer contains a parameter outside the transformer")
            entries.append(
                {
                    "name": names[id(parameter)],
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": parameter.requires_grad,
                }
            )
        groups.append(entries)
    return groups


def install_adafactor_state_restore(optimizer):
    """Restore saved moments directly, bypassing PyTorch's parameter-dtype cast.

    The normal loader still validates groups and restores their settings. Its
    casts must never become the source of the restored FP32 accumulators.
    """
    from transformers import Adafactor

    if not isinstance(optimizer, Adafactor) or getattr(optimizer, "_h3_state_restore", False):
        return
    pending = {}

    def before_load(_optimizer, state_dict):
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(_optimizer.param_groups) or any(
            len(saved["params"]) != len(current["params"]) for saved, current in zip(saved_groups, _optimizer.param_groups)
        ):
            raise ValueError("dense Adafactor checkpoint parameter groups do not match")
        pending.clear()
        for saved, current in zip(saved_groups, _optimizer.param_groups):
            for key, parameter in zip(saved["params"], current["params"]):
                state = state_dict["state"].get(key, {})
                pending[parameter] = {name: value for name, value in state.items() if isinstance(value, torch.Tensor)}

    def after_load(_optimizer):
        for parameter, tensors in pending.items():
            for name, value in tensors.items():
                _optimizer.state[parameter][name] = value.to(device=parameter.device, copy=True)
        pending.clear()

    optimizer.register_load_state_dict_pre_hook(before_load)
    optimizer.register_load_state_dict_post_hook(after_load)
    optimizer._h3_state_restore = True


class DenseStateHooks:
    def __init__(self, accelerator, network, optimizer):
        self.accelerator = accelerator
        self.network = accelerator.unwrap_model(network)
        self.transformer = self.network.transformer
        self.optimizer = getattr(optimizer, "optimizer", optimizer)
        self.pending_output = None
        self._defer_depth = 0

    def complete_pending_state(self):
        if self.pending_output is not None:
            (self.pending_output / INCOMPLETE).unlink()
            self.pending_output = None

    @contextmanager
    def defer_completion(self):
        """Include the trainer's resume cursor/manifest in the save transaction."""
        self._defer_depth += 1
        try:
            yield
        except BaseException:
            raise
        else:
            if self._defer_depth == 1:
                self.complete_pending_state()
        finally:
            self._defer_depth -= 1

    def _handled(self, model):
        return self.accelerator.unwrap_model(model) is self.network

    def _suspend(self):
        offloader = getattr(self.transformer, "offloader", None)
        if callable(getattr(offloader, "checkpoint_to_cpu", None)):
            return offloader, offloader.checkpoint_to_cpu()
        synchronize = getattr(offloader, "synchronize", None)
        if callable(synchronize):
            synchronize()
        return None, None

    def save(self, models, weights, output_dir):
        handled = [index for index, model in enumerate(models) if self._handled(model)]
        if not handled:
            return
        if len(handled) != 1:
            raise ValueError("expected exactly one dense H3 checkpoint model")
        if self.accelerator.is_main_process:
            output = Path(output_dir)
            self.pending_output = output
            # This marker survives process death and any later optimizer/RNG/
            # custom-state failure, including an overwrite of an existing state.
            (output / INCOMPLETE).write_text("Dense H3 state save has not completed.\n", encoding="utf-8")
            (output / MANIFEST).unlink(missing_ok=True)
            offloader, owners = self._suspend()
            try:
                state = logical_state(self.transformer)
                manifest = {
                    "format_version": FORMAT_VERSION,
                    "tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in state.items()},
                    "optimizer_groups": optimizer_signature(self.transformer, self.optimizer),
                }
                temporary = output / "model.safetensors.tmp"
                mem_eff_save_file(state, str(temporary), {"format": "pt", "h3_dense_state": str(FORMAT_VERSION)})
                os.replace(temporary, output / "model.safetensors")
                temporary = output / (MANIFEST + ".tmp")
                temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                os.replace(temporary, output / MANIFEST)
            finally:
                if offloader is not None:
                    offloader.checkpoint_restore_devices(owners)
        for index in reversed(handled):
            weights.pop(index)

    def load(self, models, input_dir):
        handled = [index for index, model in enumerate(models) if self._handled(model)]
        if not handled:
            return
        directory = Path(input_dir)
        if (directory / INCOMPLETE).exists():
            raise ValueError("dense H3 checkpoint is incomplete: a previous state save did not finish")
        manifest_path = directory / MANIFEST
        state = logical_state(self.transformer)
        expected = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in state.items()}
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported dense H3 state format")
            if manifest.get("tensors") != expected:
                raise ValueError("dense H3 checkpoint tensor names/shapes/dtypes do not match the current transformer")
            if manifest.get("optimizer_groups") != optimizer_signature(self.transformer, self.optimizer):
                raise ValueError("dense H3 checkpoint optimizer parameter names/order/trainability do not match")
        with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as reader:
            if manifest is None and "h3_dense_state" in (reader.metadata() or {}):
                raise ValueError("dense H3 checkpoint publication is incomplete: h3_dense_state.json is missing")
            source_names = {}
            for name in reader.keys():
                logical_name = name
                if manifest is None:
                    # Old Accelerate files serialize the dense wrapper, whereas
                    # new files serialize the transformer's canonical state.
                    logical_name = canonical_name(name).removeprefix("transformer.")
                if logical_name in source_names:
                    raise ValueError(f"dense state canonical-name collision: {logical_name}")
                source_names[logical_name] = name
            if set(source_names) != set(expected):
                raise ValueError(
                    "dense H3 checkpoint is incomplete: model tensor keys differ from the current transformer. "
                    "Legacy packed-ring states may have omitted tensors and cannot safely resume. "
                    "Use a complete matching ordinary checkpoint for a new run; omitted weights cannot be reconstructed."
                )
            # Validate the entire payload before retiring slots or writing a parameter.
            for name, target in state.items():
                source = reader.get_tensor(source_names[name])
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(f"dense H3 checkpoint shape/dtype mismatch: {name}")
                del source
            if manifest is None:
                logger.warning(
                    "Loading complete legacy dense H3 state without a named optimizer signature; "
                    "optimizer groups use the legacy positional mapping"
                )
            offloader, owners = self._suspend()
            try:
                # Suspension rebinds streamed Parameters and buffers to CPU masters.
                state = logical_state(self.transformer)
                with torch.no_grad():
                    for name, target in state.items():
                        target.copy_(reader.get_tensor(source_names[name]))
            finally:
                if offloader is not None:
                    offloader.checkpoint_restore_devices()
        for index in reversed(handled):
            models.pop(index)


def register_dense_state_hooks(accelerator, network, optimizer):
    if getattr(accelerator, "_h3_dense_state_hooks", None) is not None:
        raise ValueError("dense H3 state hooks are already registered on this accelerator")
    hooks = DenseStateHooks(accelerator, network, optimizer)
    accelerator.register_save_state_pre_hook(hooks.save)
    accelerator.register_load_state_pre_hook(hooks.load)
    original_save_state = accelerator.save_state

    @wraps(original_save_state)
    def save_complete_state(*args, **kwargs):
        hooks.pending_output = None
        result = original_save_state(*args, **kwargs)
        # Pre-hooks precede optimizer/scheduler/RNG/custom-state writes. Publish
        # completion only after Accelerate's entire save has returned normally.
        if hooks._defer_depth == 0:
            hooks.complete_pending_state()
        return result

    accelerator.save_state = save_complete_state
    accelerator._h3_dense_state_hooks = hooks
    return hooks
