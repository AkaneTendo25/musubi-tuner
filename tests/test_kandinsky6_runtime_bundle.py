import os
from pathlib import Path
import site
import subprocess
import sys

import torch


def test_bundled_runtime_imports_without_upstream_checkout(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = """
import sys
from musubi_tuner.kandinsky6.runtime.pipeline.config import PipelineConfig, load_config
from musubi_tuner.kandinsky6.runtime.pipeline.factory import create_bare_dit, get_pipeline
from musubi_tuner.kandinsky6.runtime.core.components.text_embedder import Kandinsky6TextEmbedder
from musubi_tuner.kandinsky6.runtime.core.components.vae_video import build_vae
from musubi_tuner.kandinsky6.runtime.core.components.vae_audio import build_audio_vae, build_vocoder
assert not any(name == 'kandinsky' or name.startswith('kandinsky.') for name in sys.modules)
assert not any(name == 'kandinsky_sr' or name.startswith('kandinsky_sr.') for name in sys.modules)
assert not any('kandinsky6-upstream' in path.lower() or 'kandinsky6-sr-upstream' in path.lower() for path in sys.path)
print(PipelineConfig.__name__, create_bare_dit.__name__, get_pipeline.__name__)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    dependency_paths = {
        path
        for path in [*site.getsitepackages(), site.getusersitepackages(), *sys.path]
        if path and "site-packages" in path.lower() and Path(path).is_dir()
    }
    bootstrap = (
        f"import sys; [sys.path.append(path) for path in {sorted(dependency_paths)!r} if path not in sys.path]; "
        f"sys.path.insert(0, {str(source_root)!r});\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap + code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PipelineConfig create_bare_dit get_pipeline" in result.stdout


def test_runtime_top_level_lazily_exports_pipeline_factory():
    import importlib

    module_name = "musubi_tuner.kandinsky6.runtime.pipeline.factory"
    sys.modules.pop(module_name, None)
    runtime = importlib.reload(importlib.import_module("musubi_tuner.kandinsky6.runtime"))
    assert module_name not in sys.modules
    assert runtime.get_pipeline.__name__ == "get_pipeline"
    assert module_name in sys.modules


def test_bundled_checkpoint_config_and_disabled_sr_defaults():
    from musubi_tuner.kandinsky6.runtime.pipeline.config import load_config

    runtime_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "kandinsky6" / "runtime"
    config = load_config(runtime_root / "configs" / "checkpoints" / "lite.yaml")
    assert config.checkpoint == "lite"
    assert config.dit.model_dim == 1792
    assert config.dit.model_dim_a == 896
    assert config.dit.is_multimodal is True
    assert config.sr.enabled is False


def test_runtime_bundle_records_pinned_sources_and_licenses():
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "kandinsky6" / "runtime"
    provenance = (runtime_root / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "01c857d4571c2fe9c676f19d1c2d814e66935493" in provenance
    assert "607f775d9859c026b9138966a5ecd303f60d205d" in provenance
    assert (runtime_root / "LICENSE-KANDINSKY6").is_file()
    assert (runtime_root / "sr" / "LICENSE-KANDINSKY6-SR").is_file()
    assert (runtime_root / "sr" / "LICENSE-APACHE").is_file()


def test_generation_parser_exposes_convrot_int8(monkeypatch):
    from musubi_tuner import kandinsky6_generate_video as generate

    monkeypatch.setattr(sys, "argv", ["generate", "--config", "pro", "--prompt", "x", "--output", "x.mp4"])
    assert generate.parse_args().convrot_int8 is False
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate", "--config", "pro", "--prompt", "x", "--output", "x.mp4", "--convrot_int8"],
    )
    assert generate.parse_args().convrot_int8 is True


def test_convrot_factory_uses_resolved_checkpoint_and_attaches_lora(monkeypatch):
    from types import SimpleNamespace

    from musubi_tuner import kandinsky6_generate_video as generate
    from musubi_tuner.kandinsky6.dit import TransformerEncoderBlock
    from musubi_tuner.kandinsky6.runtime.runtime.kernels import SelfAttentionEngine as RuntimeSelfAttentionEngine

    block = TransformerEncoderBlock(64, 32, 128, 64, "sdpa")
    original_attention_projection = block.attn
    original_query_projection = block.attn.to_query
    dit = torch.nn.Sequential(block)
    captured = {}

    def fake_loader(checkpoint, *, device, quant_device):
        captured.update(checkpoint=checkpoint, device=device, quant_device=quant_device)
        return dit

    monkeypatch.setattr("musubi_tuner.kandinsky6.load_dit_convrot_int8", fake_loader)
    monkeypatch.setattr(generate, "apply_lora_weights", lambda model, paths, multipliers, *, merge: captured.update(
        model=model, paths=paths, multipliers=multipliers, merge=merge
    ))
    cfg = SimpleNamespace(paths=SimpleNamespace(dit="snapshot/transformer"))
    loader = generate._convrot_dit_loader(
        enabled=True,
        compute_device="cuda:3",
        lora_paths=["adapter.safetensors"],
        lora_multipliers=[0.75],
    )
    loaded = loader(cfg, torch.device("cpu"), "sdpa")

    assert loaded is dit
    assert block.attn is original_attention_projection
    assert block.attn.to_query is original_query_projection
    assert block.attn.attn.__class__ is RuntimeSelfAttentionEngine
    assert captured == {
        "checkpoint": "snapshot/transformer",
        "device": torch.device("cpu"),
        "quant_device": "cuda:3",
        "model": dit,
        "paths": ["adapter.safetensors"],
        "multipliers": [0.75],
        "merge": False,
    }


def test_live_real_lora_round_trip_is_registered_for_pipeline_device_moves(tmp_path):
    from musubi_tuner import kandinsky6_generate_video as generate
    from musubi_tuner.kandinsky6 import DiffusionTransformer3D, LITE_CONFIG
    from musubi_tuner.networks.lora_kandinsky6 import create_arch_network

    def tiny_dit():
        config = dict(LITE_CONFIG)
        config.update(
            model_dim=64,
            model_dim_a=64,
            ff_dim=128,
            ff_dim_a=64,
            time_dim=32,
            time_dim_a=32,
            num_text_blocks=1,
            num_visual_blocks=1,
            axes_dims=(16, 24, 24),
            axes_dims_a=(16, 24, 24),
            in_text_dim=48,
            in_text_dim2=16,
        )
        return DiffusionTransformer3D(**config)

    source = tiny_dit()
    source_network = create_arch_network(1.0, 2, 2.0, None, [], source)
    source_network.apply_to(None, source, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for name, parameter in source_network.named_parameters():
            if "lora_up" in name:
                parameter.fill_(0.25)
    state = {name: value.detach().cpu() for name, value in source_network.state_dict().items()}
    assert state and any("lora_up" in name and value.count_nonzero() for name, value in state.items())
    checkpoint = tmp_path / "adapter.safetensors"
    from safetensors.torch import save_file

    save_file(state, checkpoint)
    dit = tiny_dit()
    generate.apply_lora_weights(dit, [str(checkpoint)], merge=False)

    network = dit._inference_lora_0
    assert dit._inference_lora_0 is network
    loaded_state = network.state_dict()
    assert loaded_state.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(loaded_state[name], state[name])
    dit.to(torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in network.parameters())
