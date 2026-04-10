import argparse
import logging
from pathlib import Path

import torch

from musubi_tuner.magihuman import parse_magihuman_config
from musubi_tuner.magihuman.common import EngineConfig
from musubi_tuner.magihuman.infra.checkpoint import load_model_checkpoint
from musubi_tuner.magihuman.model.dit import DiTModel
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_output_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported output dtype: {name}")


def _materialize_state_dict_dtype(model: torch.nn.Module, output_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    converted_state_dict: dict[str, torch.Tensor] = {}
    dtype_counts: dict[str, int] = {}
    for name, tensor in model.state_dict().items():
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=output_dtype)
        converted_state_dict[name] = tensor
        dtype_key = str(tensor.dtype) if hasattr(tensor, "dtype") else type(tensor).__name__
        dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + 1

    logger.info("Prepared output state dict dtypes: %s", dtype_counts)
    return converted_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MagiHuman base checkpoint from shard directory to a single safetensors file.",
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, help="Path to the source MagiHuman checkpoint directory or safetensors file.")
    parser.add_argument("--output", required=True, help="Path to the output .safetensors file.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Parameter dtype to materialize in the output checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    args, _ = parser.parse_known_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_dtype = _parse_output_dtype(args.dtype)
    config = parse_magihuman_config()
    config.arch_config.params_dtype = output_dtype
    config.arch_config.compute_dtype = output_dtype

    logger.info("Instantiating MagiHuman DiT in %s for checkpoint conversion.", args.dtype)
    model = DiTModel(model_config=config.arch_config)
    model.eval()
    model = load_model_checkpoint(model, EngineConfig(load=args.input))

    metadata = {
        "format": "musubi_tuner.magihuman.dit",
        "source": str(args.input),
        "dtype": args.dtype,
    }
    logger.info("Saving single-file checkpoint to %s", output_path)
    state_dict = _materialize_state_dict_dtype(model, output_dtype)
    mem_eff_save_file(state_dict, str(output_path), metadata=metadata)

    size_gib = output_path.stat().st_size / (1024**3)
    logger.info("Checkpoint conversion complete: %s (%.2f GiB)", output_path, size_gib)


if __name__ == "__main__":
    main()
