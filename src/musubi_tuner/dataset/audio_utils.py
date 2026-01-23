import glob
import os

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"]


def glob_audio(directory: str, base: str = "*") -> list[str]:
    audio_paths: list[str] = []
    for ext in AUDIO_EXTENSIONS:
        if base == "*":
            audio_paths.extend(
                glob.glob(os.path.join(glob.escape(directory), base + ext))
            )
        else:
            audio_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    return list(set(audio_paths))
