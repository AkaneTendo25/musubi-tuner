"""Pre-compute the Cosmos3 reasoner (und) tower K/V for a dataset.

Cosmos3 has no text encoder; the caption is consumed by the ~8B "reasoner" tower
inside the joint transformer.  That tower is a closed text-only sub-network (see
``musubi_tuner.cosmos3.reasoner_kv_cache``), so its per-layer K/V depends only on
the text token ids -- not on the noisy latent, the timestep, the resolution, or
the frame count.  Computing it once per caption lets training skip the und
parameters entirely, at a cost of roughly 144 KiB per text token on disk.

Input is the existing text-encoder cache (which stores ``varlen_input_ids_int64``
per item), so this script inherits the dataset handling already used by
``cosmos3_cache_text_encoder_outputs.py``.  Output is content-addressed by token
ids, so repeated captions are stored once.

Usage:
    python -m musubi_tuner.cosmos3_cache_reasoner_kv \
        --dataset_config dataset.toml \
        --dit /path/to/Cosmos3-Nano \
        --reasoner_kv_cache_dir /path/to/cache
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import torch
from safetensors import safe_open
from tqdm import tqdm

from musubi_tuner.cosmos3 import cosmos3_utils, reasoner_kv_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_text_encoder_cache_files(dataset_config: str) -> list[str]:
    """Collect text-encoder cache files referenced by *dataset_config*.

    Reads the dataset toml directly rather than constructing the full dataset
    group: only the cache directories are needed, and this keeps the script
    independent of dataset-class changes.
    """
    import toml

    with open(dataset_config, "r", encoding="utf-8") as handle:
        config = toml.load(handle)

    directories = []
    for dataset in config.get("datasets", []):
        cache_dir = dataset.get("cache_directory")
        if cache_dir:
            directories.append(cache_dir)
        else:
            for key in ("image_directory", "video_directory"):
                if dataset.get(key):
                    directories.append(dataset[key])

    files: list[str] = []
    for directory in directories:
        files.extend(glob.glob(os.path.join(directory, "**", "*_te.safetensors"), recursive=True))
        files.extend(glob.glob(os.path.join(directory, "**", "*_te_*.safetensors"), recursive=True))
    return sorted(set(files))


def read_input_ids(path: str) -> tuple[torch.Tensor | None, str | None]:
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        if "varlen_input_ids_int64" not in keys:
            return None, None
        ids = handle.get_tensor("varlen_input_ids_int64")
        metadata = handle.metadata() or {}
        caption = metadata.get("caption1")
    return ids, caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default=None, help="dataset toml; omit to cache only sample prompts")
    parser.add_argument("--dit", required=True, help="Path to the Cosmos3 model root")
    parser.add_argument("--transformer_subfolder", default="transformer")
    parser.add_argument("--reasoner_kv_cache_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vae_scale_factor_temporal", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    # Sample-prompt caching. These must match the values training will use, or the
    # tokenization -- and therefore the cache keys -- will not line up.
    parser.add_argument("--sample_prompts", default=None, help="also cache prompts from this file")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer_subfolder", default="text_tokenizer")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--no_system_prompt", action="store_true")
    parser.add_argument("--no_resolution_template", action="store_true")
    parser.add_argument("--no_duration_template", action="store_true")
    args = parser.parse_args()

    if not args.dataset_config and not args.sample_prompts:
        raise SystemExit("Nothing to do: pass --dataset_config and/or --sample_prompts")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    te_files = find_text_encoder_cache_files(args.dataset_config) if args.dataset_config else []
    if args.dataset_config and not te_files:
        raise SystemExit(
            "No text-encoder cache files found. Run cosmos3_cache_text_encoder_outputs.py first."
        )
    logger.info(f"Found {len(te_files)} text-encoder cache files")

    # De-duplicate by token ids before touching the GPU: many datasets repeat
    # captions, and each unique caption only needs computing once.
    pending: dict[str, tuple[torch.Tensor, str | None]] = {}
    skipped_existing = 0
    enable_fps_modulation = True  # refined from the model config once loaded

    raw: list[tuple[torch.Tensor, str | None]] = []
    for path in tqdm(te_files, desc="scanning text cache"):
        ids, caption = read_input_ids(path)
        if ids is None:
            continue
        raw.append((ids, caption))
    logger.info(f"Read {len(raw)} token sequences from the dataset")

    if args.sample_prompts:
        sample_entries = reasoner_kv_cache.sample_prompt_cache_entries(
            args, args.sample_prompts, args.vae_scale_factor_temporal
        )
        logger.info(
            f"Tokenized {len(sample_entries)} sample-prompt sequences "
            f"({len(sample_entries)//2} prompts x cond/uncond)"
        )
        raw.extend(sample_entries)

    reasoner_kv_cache.install_patches()
    logger.info(f"Loading transformer from {args.dit}")
    transformer = cosmos3_utils.load_transformer(args.dit, args.transformer_subfolder, dtype, device)
    transformer.eval()
    transformer.requires_grad_(False)
    reasoner_kv_cache.rebind_dispatch(transformer)

    config = transformer.config
    enable_fps_modulation = bool(getattr(config, "enable_fps_modulation", True))
    num_layers = sum(
        1 for m in transformer.modules() if isinstance(m, reasoner_kv_cache._unified_mot.PackedAttentionMoT)
    )
    latent_channels = int(config.latent_channel_size)

    for ids, caption in raw:
        key = reasoner_kv_cache.text_ids_cache_key(ids, enable_fps_modulation)
        if key in pending:
            continue
        if not args.overwrite and reasoner_kv_cache.cache_path_for(args.reasoner_kv_cache_dir, key).exists():
            skipped_existing += 1
            continue
        pending[key] = (ids, caption)

    logger.info(
        f"{len(pending)} unique captions to compute "
        f"({skipped_existing} already cached, {len(raw) - len(pending) - skipped_existing} duplicates)"
    )
    if not pending:
        logger.info("Nothing to do.")
        return

    # The und K/V is independent of the gen sequence, so a minimal dummy latent
    # is sufficient to drive the forward pass that produces it.
    dummy_vision = torch.zeros(1, latent_channels, 1, 8, 8, device=device, dtype=dtype)
    timestep = torch.tensor(0.5, device=device)

    total_bytes = 0
    for key, (ids, caption) in tqdm(list(pending.items()), desc="computing reasoner K/V"):
        capture = reasoner_kv_cache.ReasonerKVCapture(num_layers, store_dtype=dtype)
        packed = cosmos3_utils.build_packed_sequence(
            transformer,
            input_ids=ids.to(device),
            vision_tokens=dummy_vision,
            timestep=timestep,
            has_image_condition=False,
            fps=24.0,
            device=device,
            vae_scale_factor_temporal=args.vae_scale_factor_temporal,
        )
        with torch.no_grad():
            transformer(
                packed,
                fps_vision=torch.tensor([24.0], device=device, dtype=torch.float32),
                memory=capture,
            )
        rkv = capture.to_reasoner_kv(key)
        path = reasoner_kv_cache.save_reasoner_kv(args.reasoner_kv_cache_dir, rkv, caption=caption)
        total_bytes += os.path.getsize(path)

    logger.info(
        f"Wrote {len(pending)} reasoner K/V caches to {args.reasoner_kv_cache_dir} "
        f"({total_bytes/2**30:.2f} GiB total)"
    )


if __name__ == "__main__":
    main()
