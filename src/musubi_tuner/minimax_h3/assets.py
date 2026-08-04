from __future__ import annotations

from pathlib import Path


TEXT_ENCODER_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "text_encoder"
_REQUIRED_TEXT_ENCODER_ASSETS = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
)


def default_text_encoder_assets() -> Path:
    missing = [name for name in _REQUIRED_TEXT_ENCODER_ASSETS if not (TEXT_ENCODER_ASSET_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MiniMax H3 packaged text-encoder assets are incomplete: {', '.join(missing)}")
    return TEXT_ENCODER_ASSET_DIR
