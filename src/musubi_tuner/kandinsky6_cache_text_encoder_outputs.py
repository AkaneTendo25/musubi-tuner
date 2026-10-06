from __future__ import annotations

import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_text_batch(
    text_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert either packed or padded upstream text output into per-item varlen rows."""
    batch_size = pooled_embeds.shape[0]
    result = []
    if text_embeds.ndim == 2:
        if cu_seqlens.numel() != batch_size + 1:
            raise ValueError("Packed Kandinsky6 text output requires one cumulative length per item")
        for index in range(batch_size):
            start, end = int(cu_seqlens[index]), int(cu_seqlens[index + 1])
            rows = text_embeds[start:end]
            result.append((rows, pooled_embeds[index], torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)))
        return result
    if text_embeds.ndim != 3 or text_embeds.shape[0] != batch_size:
        raise ValueError(f"Unexpected Kandinsky6 text tensor shape {tuple(text_embeds.shape)}")
    for index in range(batch_size):
        mask = (
            torch.ones(text_embeds.shape[1], dtype=torch.bool, device=text_embeds.device)
            if attention_mask is None
            else attention_mask[index].bool()
        )
        rows = text_embeds[index, mask]
        result.append((rows, pooled_embeds[index], torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)))
    return result


@torch.no_grad()
def encode_and_save_batch(text_embedder, batch: list[ItemInfo]) -> None:
    from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_kandinsky6

    outputs, cu_seqlens, attention_mask = text_embedder.encode([item.caption for item in batch])
    rows = split_text_batch(outputs["text_embeds"], outputs["pooled_embed"], cu_seqlens, attention_mask)
    for item, (text_embed, pooled_embed, mask) in zip(batch, rows):
        save_text_encoder_output_cache_kandinsky6(
            item,
            text_embeds=text_embed.cpu(),
            pooled_embed=pooled_embed.cpu(),
            attention_mask=mask.cpu(),
        )


def main() -> None:
    parser = cache_text.setup_parser_common()
    parser.add_argument("--text_encoder_qwen", required=True, help="Qwen2.5-VL text encoder directory")
    parser.add_argument("--text_encoder_clip", required=True, help="CLIP text encoder directory")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--quantized_qwen", action="store_true")
    parser.add_argument("--upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    parser.add_argument("--sr_upstream", help="deprecated compatibility option; ignored because the runtime is bundled")
    args = parser.parse_args()

    try:
        from musubi_tuner.kandinsky6.runtime.core.components.text_embedder import Kandinsky6TextEmbedder
    except ImportError as exc:
        raise RuntimeError(
            "Kandinsky6 text caching requires the optional Kandinsky6 dependencies; install `musubi-tuner[kandinsky6]`."
        ) from exc
    from musubi_tuner.dataset.architectures import ARCHITECTURE_KANDINSKY6

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config), args, architecture=ARCHITECTURE_KANDINSKY6
    )
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    existing, expected = cache_text.prepare_cache_files_and_paths(datasets)
    embedder = Kandinsky6TextEmbedder(
        args.text_encoder_qwen,
        args.text_encoder_clip,
        max_length=args.max_length,
        device=device,
        quantized_qwen=args.quantized_qwen,
        text_token_padding=False,
    )
    cache_text.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        existing,
        expected,
        lambda batch: encode_and_save_batch(embedder, batch),
    )
    cache_text.post_process_cache_files(datasets, existing, expected, args.keep_cache)


if __name__ == "__main__":
    main()
