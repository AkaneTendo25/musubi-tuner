import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MagiHuman inference wrapper for musubi-tuner")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation.")
    parser.add_argument("--image_path", type=str, required=True, help="Reference image path.")
    parser.add_argument("--audio_path", type=str, default=None, help="Optional audio path.")
    parser.add_argument("--config_load_path", type=str, default=None, help="Vendored MagiHuman config JSON.")
    parser.add_argument("--save_path_prefix", type=str, required=True, help="Output path prefix.")
    return parser.parse_args()


def main():
    parse_args()
    raise NotImplementedError("Standalone MagiHuman video generation is not available in this package.")


if __name__ == "__main__":
    main()
