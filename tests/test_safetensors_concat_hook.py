import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from yue2_fakes import write_safetensors_ordered  # noqa: E402

from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8  # noqa: E402
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks  # noqa: E402
from musubi_tuner.yue2.yue2_checkpoint import CheckpointLayout, build_weight_transform_hooks  # noqa: E402
from musubi_tuner.yue2.yue2_model import YuE2Config  # noqa: E402

ROLES = ("q", "k", "v")


def _parts():
    g = torch.Generator().manual_seed(0)
    return {r: torch.randn(3 + i, 8, generator=g) for i, r in enumerate(ROLES)}


def _hooks():
    def split(key, tensor):
        if key.startswith("attn.") and key.split(".")[1] in ROLES:
            return None, None
        return [key], (None if tensor is None else [tensor])

    def concat(key, tensors):
        if tensors is None:
            return "attn.qkv.weight", None
        by_role = {k.split(".")[1]: t for k, t in tensors.items()}
        return "attn.qkv.weight", torch.cat([by_role[r] for r in ROLES])

    return WeightTransformHooks(split_hook=split, concat_hook=concat)


@pytest.mark.parametrize("order", [("k", "q", "v"), ("v", "k", "q"), ("q", "k", "v")])
def test_concat_key_listed_once_rows_by_role(tmp_path, order):
    parts = _parts()
    tensors = {"other.weight": torch.ones(2, 2)}
    tensors.update({f"attn.{r}.weight": parts[r] for r in order})
    path = write_safetensors_ordered(tmp_path / "m.safetensors", tensors)
    with MemoryEfficientSafeOpen(path) as f:
        assert f.keys() == list(tensors)  # the header keeps the written order
        adapter = TensorWeightAdapter(_hooks(), f)
        assert adapter.keys() == ["other.weight", "attn.qkv.weight"]
        fused = adapter.get_tensor("attn.qkv.weight")
    assert torch.equal(fused, torch.cat([parts["q"], parts["k"], parts["v"]]))


def test_merge_hook_runs_once_per_concatenated_key(tmp_path):
    parts = _parts()
    path = write_safetensors_ordered(tmp_path / "m.safetensors", {f"attn.{r}.weight": parts[r] for r in ("k", "q", "v")})
    down, up = torch.randn(2, 8), torch.randn(12, 2)
    lora = {
        "lora_unet_attn_qkv.lora_down.weight": down,
        "lora_unet_attn_qkv.lora_up.weight": up,
        "lora_unet_attn_qkv.alpha": torch.tensor(2.0),
    }
    sd = load_safetensors_with_lora_and_fp8(path, [lora], [1.0], False, torch.device("cpu"), weight_transform_hooks=_hooks())
    base = torch.cat([parts["q"], parts["k"], parts["v"]])
    assert list(sd) == ["attn.qkv.weight"]
    assert torch.allclose(sd["attn.qkv.weight"], base + up @ down, atol=1e-6)


def test_yue2_hf_hooks_with_shuffled_header(tmp_path):
    cfg = YuE2Config.tiny(num_layers=1)
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(n, cfg.hidden_size, generator=g) for n in (cfg.q_dim, cfg.kv_dim, cfg.kv_dim))
    gate, up = (torch.randn(cfg.intermediate_size, cfg.hidden_size, generator=g) for _ in range(2))
    p = "model.layers.0."
    tensors = {
        p + "nar_mlp.up_proj.weight": up,
        p + "self_attn.v_proj.weight": v,
        p + "nar_mlp.gate_proj.weight": gate,
        p + "self_attn.k_proj.weight": k,
        p + "self_attn.q_proj.weight": q,
        p + "nar_input_layernorm.weight": torch.ones(cfg.hidden_size),
    }
    path = write_safetensors_ordered(tmp_path / "hf.safetensors", tensors)
    with MemoryEfficientSafeOpen(path) as f:
        adapter = TensorWeightAdapter(build_weight_transform_hooks(CheckpointLayout.HF, cfg), f)
        keys = adapter.keys()
        assert sorted(keys) == sorted(
            ["nar.blocks.0.mlp.gate_up_proj.weight", "ar.blocks.0.self_attn.qkv_proj.weight", "nar.blocks.0.input_layernorm.weight"]
        )
        assert len(keys) == len(set(keys))
        assert torch.equal(adapter.get_tensor("ar.blocks.0.self_attn.qkv_proj.weight"), torch.cat([q, k, v]))
        assert torch.equal(adapter.get_tensor("nar.blocks.0.mlp.gate_up_proj.weight"), torch.cat([gate, up]))


def test_yue2_hf_hooks_reject_unknown_and_missing(tmp_path):
    cfg = YuE2Config.tiny(num_layers=1)
    path = write_safetensors_ordered(tmp_path / "bad.safetensors", {"model.layers.3.input_layernorm.weight": torch.ones(4)})
    with MemoryEfficientSafeOpen(path) as f:
        with pytest.raises(ValueError, match="num_layers"):
            TensorWeightAdapter(build_weight_transform_hooks(CheckpointLayout.HF, cfg), f)
    path = write_safetensors_ordered(tmp_path / "missing.safetensors", {"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4)})
    with MemoryEfficientSafeOpen(path) as f:
        adapter = TensorWeightAdapter(build_weight_transform_hooks(CheckpointLayout.HF, cfg), f)
        with pytest.raises(ValueError, match="missing parts"):
            adapter.get_tensor("ar.blocks.0.self_attn.qkv_proj.weight")
