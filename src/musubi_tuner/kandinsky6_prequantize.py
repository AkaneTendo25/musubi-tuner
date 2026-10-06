from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from musubi_tuner.kandinsky6.model import _resolve_weight_file, inspect_checkpoint
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file


TARGET_LAYER_KEYS = (
    "visual_transformer_blocks.",
    "video_text_transformer_blocks.",
    "audio_text_transformer_blocks.",
)
EXCLUDE_LAYER_KEYS = ("modulation", "norm")
ALLOWED_GROUPSIZES = (256, 64)


@dataclass(frozen=True)
class ExportReport:
    source: Path
    output: Path
    variant: str
    checkpoint_format: str
    tensor_count: int
    quantized_layer_count: int
    groupsize_counts: dict[int, int]
    estimated_tensor_bytes: int
    output_bytes: int


def _comfy_quant_tensor(groupsize: int) -> torch.Tensor:
    spec = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": groupsize,
    }
    return torch.tensor(list(json.dumps(spec, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def export_prequantized_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    quant_device: str | torch.device = "cuda",
    overwrite: bool = False,
    disable_numpy_memmap: bool = False,
) -> ExportReport:
    source_path = _resolve_weight_file(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("Source and output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (pass --overwrite to replace it)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variant, n_grid = inspect_checkpoint(source_path)
    if n_grid is not None:
        raise ValueError(f"Expected a regular Kandinsky 6 Lite/Pro checkpoint, got PiFlow n_grid={n_grid}")
    checkpoint_format = "regular"
    quantizer = ConvRotInt8Quantizer(
        target_layer_keys=list(TARGET_LAYER_KEYS),
        exclude_layer_keys=list(EXCLUDE_LAYER_KEYS),
        allowed_groupsizes=ALLOWED_GROUPSIZES,
    )
    state = quantizer.load_and_quantize(
        [str(source_path)],
        calc_device=torch.device(quant_device),
        move_to_device=False,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    exported: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(".scale_weight"):
            exported[key[: -len(".scale_weight")] + ".weight_scale"] = value.float().cpu()
        elif value.is_floating_point():
            exported[key] = value.to(device="cpu", dtype=torch.bfloat16)
        else:
            exported[key] = value.cpu()
    for module_path, groupsize in quantizer.module_groupsizes.items():
        exported[module_path + ".comfy_quant"] = _comfy_quant_tensor(groupsize)

    groupsize_counts = {groupsize: 0 for groupsize in ALLOWED_GROUPSIZES}
    for groupsize in quantizer.module_groupsizes.values():
        groupsize_counts[groupsize] = groupsize_counts.get(groupsize, 0) + 1
    groupsize_counts = {key: value for key, value in groupsize_counts.items() if value}
    estimated_tensor_bytes = sum(t.numel() * t.element_size() for t in exported.values())
    metadata = {
        "kandinsky6.source": str(source_path),
        "kandinsky6.variant": variant,
        "kandinsky6.checkpoint_format": checkpoint_format,
        "kandinsky6.quantization": "convrot_int8_comfy",
        "kandinsky6.tensor_count": str(len(exported)),
        "kandinsky6.quantized_layer_count": str(len(quantizer.module_groupsizes)),
        "kandinsky6.groupsize_counts": json.dumps(groupsize_counts, sort_keys=True),
        "kandinsky6.estimated_tensor_bytes": str(estimated_tensor_bytes),
    }

    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        mem_eff_save_file(exported, str(temporary_path), metadata)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return ExportReport(
        source=source_path,
        output=output_path,
        variant=variant,
        checkpoint_format=checkpoint_format,
        tensor_count=len(exported),
        quantized_layer_count=len(quantizer.module_groupsizes),
        groupsize_counts=groupsize_counts,
        estimated_tensor_bytes=estimated_tensor_bytes,
        output_bytes=output_path.stat().st_size,
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Kandinsky 6 DiT as one ComfyUI ConvRot INT8 safetensors file.")
    parser.add_argument("source", help="Regular Lite/Pro Kandinsky 6 checkpoint file or model directory")
    parser.add_argument("output", help="Destination .safetensors file")
    parser.add_argument("--quant-device", default="cuda", help="Device used for quantization (default: cuda)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination atomically")
    parser.add_argument("--disable-numpy-memmap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = setup_parser().parse_args(argv)
    report = export_prequantized_checkpoint(
        args.source,
        args.output,
        quant_device=args.quant_device,
        overwrite=args.overwrite,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )
    print(
        json.dumps(
            {
                "source": str(report.source),
                "output": str(report.output),
                "variant": report.variant,
                "checkpoint_format": report.checkpoint_format,
                "tensor_count": report.tensor_count,
                "quantized_layer_count": report.quantized_layer_count,
                "groupsize_counts": report.groupsize_counts,
                "estimated_tensor_bytes": report.estimated_tensor_bytes,
                "output_bytes": report.output_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
