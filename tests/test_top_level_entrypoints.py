from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ideogram4_top_level_entrypoints_exist():
    expected = {
        "ideogram4_cache_latents.py": "musubi_tuner.ideogram4_cache_latents",
        "ideogram4_cache_text_encoder_outputs.py": "musubi_tuner.ideogram4_cache_text_encoder_outputs",
        "ideogram4_generate_image.py": "musubi_tuner.ideogram4_generate_image",
        "ideogram4_train_network.py": "musubi_tuner.ideogram4_train_network",
    }

    for script_name, module_name in expected.items():
        script = ROOT / script_name
        assert script.exists(), f"missing top-level entrypoint: {script_name}"
        assert script.read_text(encoding="utf-8") == (f'from {module_name} import main\n\nif __name__ == "__main__":\n    main()\n')


def test_minimax_h3_latent_cache_entrypoint_exists():
    script = ROOT / "minimax_h3_cache_latents.py"

    assert script.exists(), "missing top-level entrypoint: minimax_h3_cache_latents.py"
    assert script.read_text(encoding="utf-8") == (
        'from musubi_tuner.minimax_h3_cache_latents import main\n\nif __name__ == "__main__":\n    main()\n'
    )


def test_minimax_h3_text_cache_entrypoint_exists():
    script = ROOT / "minimax_h3_cache_text_encoder_outputs.py"

    assert script.exists(), "missing top-level entrypoint: minimax_h3_cache_text_encoder_outputs.py"
    assert script.read_text(encoding="utf-8") == (
        'from musubi_tuner.minimax_h3_cache_text_encoder_outputs import main\n\nif __name__ == "__main__":\n    main()\n'
    )


def test_minimax_h3_training_entrypoint_exists():
    script = ROOT / "minimax_h3_train_network.py"

    assert script.exists(), "missing top-level entrypoint: minimax_h3_train_network.py"
    assert script.read_text(encoding="utf-8") == (
        'from musubi_tuner.minimax_h3_train_network import main\n\nif __name__ == "__main__":\n    main()\n'
    )


def test_minimax_h3_training_module_is_directly_executable():
    module = ROOT / "src" / "musubi_tuner" / "minimax_h3_train_network.py"

    assert module.read_text(encoding="utf-8").endswith('\n\nif __name__ == "__main__":\n    main()\n')


def test_minimax_h3_generation_entrypoint_exists():
    script = ROOT / "minimax_h3_generate_video.py"

    assert script.exists(), "missing top-level entrypoint: minimax_h3_generate_video.py"
    assert script.read_text(encoding="utf-8") == (
        'from musubi_tuner.minimax_h3_generate_video import main\n\nif __name__ == "__main__":\n    main()\n'
    )


def test_yue2_top_level_entrypoints_exist():
    expected = {
        "yue2_cache_latents.py": "musubi_tuner.yue2_cache_latents",
        "yue2_cache_text_encoder_outputs.py": "musubi_tuner.yue2_cache_text_encoder_outputs",
        "yue2_train_network.py": "musubi_tuner.yue2_train_network",
        "yue2_generate_music.py": "musubi_tuner.yue2_generate_music",
        "yue2_convert_lora.py": "musubi_tuner.networks.convert_yue2_lora",
    }

    for script_name, module_name in expected.items():
        script = ROOT / script_name
        assert script.exists(), f"missing top-level entrypoint: {script_name}"
        assert script.read_text(encoding="utf-8") == (f'from {module_name} import main\n\nif __name__ == "__main__":\n    main()\n')


def test_yue2_modules_are_directly_executable():
    modules = [
        "yue2_cache_latents.py",
        "yue2_cache_text_encoder_outputs.py",
        "yue2_train_network.py",
        "yue2_generate_music.py",
        "networks/convert_yue2_lora.py",
    ]

    for name in modules:
        module = ROOT / "src" / "musubi_tuner" / name
        assert module.read_text(encoding="utf-8").endswith('\n\nif __name__ == "__main__":\n    main()\n'), name
