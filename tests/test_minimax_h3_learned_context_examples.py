from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "minimax_h3" / "learned_context" / "generate_bootstrap_targets.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("h3_learned_context_bootstrap_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_example_generates_targets_and_scene_only_captions(tmp_path, monkeypatch) -> None:
    module = _load_example_module()
    output = tmp_path / "targets"
    manifest = tmp_path / "bootstrap.toml"
    manifest.write_text(
        f"""
[bootstrap]
output_directory = {str(output)!r}
concept = "purple water fills the scene"

[[scenes]]
name = "workshop"
caption = "A craftsperson stands in a workshop."
seed = 7

[[scenes]]
name = "museum"
caption = "Visitors cross a museum gallery."
seed = 8
""",
        encoding="utf-8",
    )
    assets = {}
    for name in ("model", "text_encoder", "vae", "audio_vae"):
        assets[name] = tmp_path / f"{name}.safetensors"
        assets[name].touch()

    prompts = []

    class Generator:
        def generate(self, request) -> None:
            prompts.append(request.prompt)
            request.output.write_bytes(b"video")

    monkeypatch.setattr(module, "generator_from_args", lambda _args, _request: Generator())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE),
            "--bootstrap_config",
            str(manifest),
            "--model",
            str(assets["model"]),
            "--text_encoder",
            str(assets["text_encoder"]),
            "--vae",
            str(assets["vae"]),
            "--audio_vae",
            str(assets["audio_vae"]),
        ],
    )

    module.main()

    assert prompts == [
        "A craftsperson stands in a workshop. purple water fills the scene",
        "Visitors cross a museum gallery. purple water fills the scene",
    ]
    assert (output / "workshop.mp4").read_bytes() == b"video"
    assert (output / "museum.mp4").read_bytes() == b"video"
    assert (output / "workshop.txt").read_text(encoding="utf-8") == "A craftsperson stands in a workshop.\n"
    assert (output / "museum.txt").read_text(encoding="utf-8") == "Visitors cross a museum gallery.\n"
