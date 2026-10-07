from types import SimpleNamespace
import inspect
import sys

import pytest
import torch

from musubi_tuner import kandinsky6_generate_video as generate
from musubi_tuner.kandinsky6.defaults import DEFAULT_NEGATIVE_PROMPT
from musubi_tuner.kandinsky6.runtime.pipeline.pipeline import Kandinsky6Pipeline
from musubi_tuner.kandinsky6_train_network import Kandinsky6NetworkTrainer


def test_pipeline_and_cli_share_the_official_negative_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate", "--config", "pro", "--prompt", "test", "--output", "clip.mp4"])
    assert generate.parse_args().negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert inspect.signature(Kandinsky6Pipeline.__call__).parameters["negative_text"].default == DEFAULT_NEGATIVE_PROMPT


@pytest.mark.parametrize("negative", [None, "", "custom negative prompt"])
def test_generation_forwards_default_or_explicit_negative(monkeypatch, tmp_path, negative):
    seen = {}

    class Pipeline:
        dit = torch.nn.Linear(1, 1)

        def __call__(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(frames=torch.zeros(1, 3, 1, 1, 1), audio=None, path=kwargs["save_path"])

    monkeypatch.setattr(generate, "_load_pipeline_factory", lambda: lambda *args, **kwargs: Pipeline())
    argv = ["generate", "--config", "pro", "--prompt", "test", "--output", str(tmp_path / "clip.mp4")]
    if negative is not None:
        argv.extend(["--negative_prompt", negative])
    monkeypatch.setattr(sys, "argv", argv)
    generate.main()
    assert seen["negative_text"] == (DEFAULT_NEGATIVE_PROMPT if negative is None else negative)


@pytest.mark.parametrize("negative", [None, "", "custom negative prompt"])
def test_training_preview_uses_same_default_and_preserves_overrides(monkeypatch, tmp_path, negative):
    seen = []

    class EncodingComplete(Exception):
        pass

    class TextEncoder:
        def to(self, device):
            return self

        def encode(self, captions):
            seen.append(captions)
            if len(seen) == 2:
                raise EncodingComplete
            return {}, None, None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("musubi_tuner.kandinsky6_train_network.should_sample_images", lambda *args: True)
    monkeypatch.setattr("musubi_tuner.kandinsky6_train_network.clean_memory_on_device", lambda *args: None)
    transformer = torch.nn.Linear(1, 1)
    transformer.switch_block_swap_for_inference = lambda: None
    transformer.switch_block_swap_for_training = lambda: None
    resources = SimpleNamespace(text_embedder=TextEncoder(), to=lambda device: None)
    accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
    sample = {"prompt": "test"}
    if negative is not None:
        sample["negative_prompt"] = negative
    trainer = Kandinsky6NetworkTrainer()
    trainer.default_guidance_scale = 5.0
    with pytest.raises(EncodingComplete):
        trainer.sample_images(
            accelerator, SimpleNamespace(output_dir=str(tmp_path)), None, 1, resources, transformer, [sample], torch.float32
        )
    assert seen[1] == [DEFAULT_NEGATIVE_PROMPT if negative is None else negative]
