# LTXV2 / LTXAV Training (Current State)

This document describes the current state of the **LTXV2** training pipeline implemented in this repository.

- Trainer entry point: `ltxv2_train_network.py` (repo root)
- Implementation: `src/musubi_tuner/ltxv2_train_network.py`

LTXV2 now uses the vendored official **LTX-2** sources (under `src/musubi_tuner/ltxv2/ltx-core`) via a musubi-compatible adapter:

- Adapter: `src/musubi_tuner/networks/lora_ltxv2.py` (`OfficialLTXV2Wrapper`)

## Supported modalities

The modality is selected via `--ltx_mode`:

- `video`
  - Video-only training (LTXV)
- `av`
  - Audio-video training (LTXAV)
  - Requires audio latent sidecar cache files
- `audio`
  - Reserved for future audio-only training

`--ltxv2_audio_video` is still present for backward compatibility; if `--ltx_mode` is not specified, it may derive `av` from this flag.

## Required caching workflow

Training expects **cached** data on disk (latents and text embeddings) produced by the dedicated caching scripts.

### 1) Cache video latents

Script: `ltxv2_cache_latents.py`

This produces per-item latent cache files containing keys like:

- `latents_{F}x{H}x{W}_{dtype}`

At batch assembly time, these become `batch["latents"]`.

### 2) Cache audio latents (AV only)

Script: `ltxv2_cache_audio_latents.py`

Implementation module: `src/musubi_tuner/ltxv2_cache_audio_latents.py`

This writes **sidecar** cache files next to the video latent cache:

- `*_ltxv2_audio.safetensors`

Keys currently written by the stub cache:

- `audio_latents_{F}x1x1_{dtype}`
- `audio_lengths_{int_dtype}`

At batch assembly time, these become:

- `batch["audio_latents"]`
- `batch["audio_lengths"]`

Audio caches are loaded automatically by the dataset batch manager when the sidecar file exists.

### 3) Cache text encoder outputs

Script: `ltxv2_cache_text_encoder_outputs.py`

It writes:

- `text_{dtype}`
- `text_mask`

At batch assembly time, these become:

- `batch["text"]`
- `batch["text_mask"]`

For AV (`--ltxv2_audio_video` in the caching script), the cached `text` embedding is concatenated as `[video_ctx, audio_ctx]` along the last dimension.
`OfficialLTXV2Wrapper` will automatically split this into per-modality contexts.

## Vendored package keep list

Only the following vendored modules are required by the musubi pipeline (cache latents → cache text → train).

### ltx-core (required)

- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/types.py`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/components/patchifiers.py`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/guidance/perturbations.py`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/loader/sft_loader.py`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/model/transformer/**`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/model/video_vae/**`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/model/audio_vae/**` (AV only)
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/model/common/**`
- `src/musubi_tuner/ltxv2/ltx-core/src/ltx_core/text_encoders/gemma/**` (only if using official Gemma caching)

### ltx-trainer (optional)

- `src/musubi_tuner/ltxv2/ltx-trainer/src/ltx_trainer/model_loader.py` (convenience component loaders)

### ltx-pipelines (optional)

- Not required for training. Keep only if you want their I/O / inference helpers.

### How caches are merged into a batch

The dataset batch manager merges multiple `.safetensors` sources per item:

1. video latent cache (`item_info.latent_cache_path`)
2. optional audio latent cache sidecar (`item_info.audio_latent_cache_path`)
3. text encoder output cache (`item_info.text_encoder_output_cache_path`)

Suffixes like dtype and `{F}x{H}x{W}` are stripped when building the batch dict, so training sees stable keys such as `latents`, `text`, `text_mask`, `audio_latents`.

## Training command (canonical)

Use `accelerate` for training.

Video-only training uses `--ltx_mode video`.

Audio-video training uses `--ltx_mode av`.

### Required explicit timestep format

**You must set `--ltxv2_timestep_format` explicitly.**

Supported values:

- `flowmatch_1_1000`
- `legacy_0_1000`
- `sd3_0_1`

The trainer normalizes the provided timesteps to the model’s expected `0..1` range.

### Example: video-only (T5)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltxv2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode video \
  --ltxv2_backend official \
  --ltxv2_model /path/to/ltxv2_dit.safetensors \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --text_encoder google/t5-v1_1-xxl \
  --text_encoder_backend t5 \
  --tokenizer google/t5-v1_1-xxl \
  --max_length 256 \
  --ltxv2_timestep_format flowmatch_1_1000 \
  --network_module musubi_tuner.networks.lora_ltxv2 \
  --network_dim 16 \
  --network_alpha 16
```

### Example: video-only (Gemma)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltxv2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode video \
  --ltxv2_backend official \
  --ltxv2_model /path/to/ltxv2_dit.safetensors \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --text_encoder google/gemma-3-12b-it \
  --text_encoder_backend gemma \
  --tokenizer google/gemma-3-12b-it \
  --max_length 256 \
  --ltxv2_timestep_format flowmatch_1_1000 \
  --network_module musubi_tuner.networks.lora_ltxv2 \
  --network_dim 16 \
  --network_alpha 16
```

### Example: audio-video (AV)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltxv2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode av \
  --ltxv2_backend official \
  --ltxv2_model /path/to/ltxav_dit.safetensors \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --text_encoder google/gemma-3-12b-it \
  --text_encoder_backend gemma \
  --tokenizer google/gemma-3-12b-it \
  --max_length 256 \
  --ltxv2_timestep_format flowmatch_1_1000 \
  --network_module musubi_tuner.networks.lora_ltxv2 \
  --network_dim 16 \
  --network_alpha 16
```

## Preview sampling (`--sample_prompts`)

If `--sample_prompts` is provided, the trainer will:

- Load tokenizer and text encoder according to `--text_encoder_backend`
- Encode prompt(s) to hidden states
- Run a FlowMatchDiscreteScheduler denoising loop
- Decode latents with the VAE and save preview output

The preview path uses the same `--ltxv2_timestep_format` normalization as training.

## Loss weighting

LTXV2 supports the following loss weights:

- `--video_loss_weight` (applied in `call_dit` by scaling both prediction and target by `sqrt(weight)` so MSE is effectively multiplied by `weight`)
- `--audio_loss_weight` (reserved; audio loss is not implemented yet)

## Cache key compatibility

Preferred cached text keys:

- `text` / `text_mask`

Legacy keys:

- `t5` / `t5_mask`

If legacy keys are detected during training, the trainer emits a one-time warning.

## Known limitations

- `--ltx_mode audio`
  - Not implemented.
- AV inference previews
  - The current preview path is video-only; audio preview generation is not implemented.

## Troubleshooting

- If training fails with missing cache keys:
  - Ensure you ran both latent caching and text caching.
  - For AV, ensure audio sidecar caches exist and are discoverable.
- If embeddings shape mismatches occur:
  - Ensure the text caching backend and training backend match (`t5` vs `gemma`).
  - For AV, ensure you cached text with `--ltxv2_audio_video` so embeddings are concatenated.
