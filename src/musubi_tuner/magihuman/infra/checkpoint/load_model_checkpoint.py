# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import os
import subprocess

import torch
from safetensors.torch import load as load_from_bytes
from tqdm.auto import tqdm

from ...common import EngineConfig
from ...utils import print_rank_0
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors


def _load_shard(shard_path, param_names, num_threads=None):
    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        cmd = ["zstd", "-d"]
        if num_threads:
            cmd.extend(["-T", str(num_threads)])  # set parallelism

        process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)

        decompressed_data = process.stdout.read()
        while True:
            new_data = process.stdout.read()
            if not new_data:
                break
            decompressed_data += new_data
        process.stdout.close()

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Decompression failed: {process.stderr.read().decode()}")

        buffer = io.BytesIO(decompressed_data)
        weights = load_from_bytes(buffer.getvalue())
        buffer.close()
    else:
        weights = load_safetensors(shard_path, device="cpu", disable_mmap=True, disable_numpy_memmap=True)

    if param_names is None:
        return weights
    return {name: weights[name] for name in param_names}


def _iter_safetensors_items(
    shard_path: str,
    param_names=None,
    device: str | torch.device | None = "cpu",
    dtype=None,
    disable_numpy_memmap: bool = False,
):
    target_device = torch.device(device) if device is not None else None
    with MemoryEfficientSafeOpen(shard_path, disable_numpy_memmap=disable_numpy_memmap) as handle:
        keys = param_names if param_names is not None else list(handle.keys())
        for name in keys:
            yield name, handle.get_tensor(name, device=target_device, dtype=dtype)


def _copy_state_dict_items_into_model(model, items):
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    loaded_keys = set()
    unexpected_keys = []

    with torch.no_grad():
        for param_name, tensor in items:
            target_tensor = model_state_dict.get(param_name)
            if target_tensor is None:
                unexpected_keys.append(param_name)
                continue

            if tuple(target_tensor.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"MagiHuman checkpoint tensor shape mismatch for '{param_name}': "
                    f"checkpoint {tuple(tensor.shape)} vs model {tuple(target_tensor.shape)}"
                )

            if tensor.dtype != target_tensor.dtype or tensor.device != target_tensor.device:
                tensor = tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)

            target_tensor.copy_(tensor)
            loaded_keys.add(param_name)

    return loaded_keys, unexpected_keys, model_keys


def _load_state_dict_fast(
    model,
    shard_path: str,
    device: str | torch.device | None,
    dtype,
    disable_numpy_memmap: bool,
):
    state_dict = load_safetensors(
        shard_path,
        device=device,
        disable_mmap=True,
        dtype=dtype,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    missing_keys = sorted(info.missing_keys)
    unexpected_keys = sorted(info.unexpected_keys)
    return missing_keys, unexpected_keys


def _iter_checkpoint_shards(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        yield checkpoint_dir, None
        return

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"MagiHuman checkpoint path does not contain model.safetensors: {checkpoint_dir}")
        yield model_file_path, None
        return

    with open(index_path, "r") as f:
        index = json.load(f)

    shard_map = {}

    # Group parameters by shard file
    for param_name, shard_file in index["weight_map"].items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        if shard_path not in shard_map:
            shard_map[shard_path] = []
        shard_map[shard_path].append(param_name)

    for shard_path, param_names in shard_map.items():
        yield shard_path, param_names


def resolve_checkpoint_weight_files(engine_config: EngineConfig) -> list[str]:
    files: list[str] = []
    for shard_path, _ in _iter_checkpoint_shards(engine_config.load):
        files.append(shard_path)
    return files


def load_model_checkpoint_state_dict(
    engine_config: EngineConfig,
    device: str | torch.device | None = "cpu",
    dtype=None,
    disable_numpy_memmap: bool = False,
):
    state_dict = {}
    for shard_path, param_names in _iter_checkpoint_shards(engine_config.load):
        if os.path.exists(shard_path + ".zst"):
            shard_state_dict = _load_shard(shard_path, param_names)
        else:
            shard_state_dict = load_safetensors(
                shard_path,
                device=device,
                disable_mmap=True,
                dtype=dtype,
                disable_numpy_memmap=disable_numpy_memmap,
            )
            if param_names is not None:
                shard_state_dict = {name: shard_state_dict[name] for name in param_names}
        state_dict.update(shard_state_dict)
    return state_dict


def load_model_checkpoint(
    model,
    engine_config: EngineConfig,
    device: str | None = "cpu",
    dtype=None,
    disable_numpy_memmap: bool = False,
    prefer_full_state_dict: bool = False,
):
    print_rank_0(f"Loading checkpoint with safetensors format from {engine_config.load}")
    shard_entries = list(_iter_checkpoint_shards(engine_config.load))
    if (
        prefer_full_state_dict
        and len(shard_entries) == 1
        and shard_entries[0][1] is None
        and not os.path.exists(shard_entries[0][0] + ".zst")
    ):
        shard_path, _ = shard_entries[0]
        print_rank_0("Using full state_dict assign=True load path for single-file MagiHuman checkpoint")
        missing_keys, unexpected_keys = _load_state_dict_fast(
            model,
            shard_path,
            device=device,
            dtype=dtype,
            disable_numpy_memmap=disable_numpy_memmap,
        )
        print_rank_0(f"Load Weight Missing Keys: {missing_keys}")
        print_rank_0(f"Load Weight Unexpected Keys: {unexpected_keys}")
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "MagiHuman checkpoint does not match the current DiT implementation. "
                f"missing keys ({len(missing_keys)}): {missing_keys} "
                f"unexpected keys ({len(unexpected_keys)}): {unexpected_keys}"
            )
        print_rank_0("Load checkpoint successfully")
        return model

    print_rank_0("Using streaming checkpoint load path for MagiHuman")
    loaded_keys = set()
    unexpected_keys = []
    model_keys = set(model.state_dict().keys())
    for shard_path, param_names in tqdm(shard_entries, desc="Loading shards", total=len(shard_entries)):
        if os.path.exists(shard_path + ".zst"):
            shard_state_dict = _load_shard(shard_path, param_names)
            shard_loaded_keys, shard_unexpected, _ = _copy_state_dict_items_into_model(model, shard_state_dict.items())
            del shard_state_dict
        else:
            shard_loaded_keys, shard_unexpected, _ = _copy_state_dict_items_into_model(
                model,
                _iter_safetensors_items(
                    shard_path,
                    param_names=param_names,
                    device=device,
                    dtype=dtype,
                    disable_numpy_memmap=disable_numpy_memmap,
                ),
            )

        loaded_keys.update(shard_loaded_keys)
        unexpected_keys.extend(shard_unexpected)

    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(set(unexpected_keys))
    print_rank_0(f"Load Weight Missing Keys: {missing_keys}")
    print_rank_0(f"Load Weight Unexpected Keys: {unexpected_keys}")
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "MagiHuman checkpoint does not match the current DiT implementation. "
            f"missing keys ({len(missing_keys)}): {missing_keys} "
            f"unexpected keys ({len(unexpected_keys)}): {unexpected_keys}"
        )
    print_rank_0("Load checkpoint successfully")
    return model
