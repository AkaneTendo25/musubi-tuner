# LTXV2 / LTXAV Training

This document describes the **LTXV2** (video) and **LTXAV** (audio-video) training pipeline implemented in this repository.

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

Keys written by the audio cache:

- `audio_latents_{T}x{F}x{C}_{dtype}`
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

## Dataset + cache directory structure

Training is designed around a dataset config (`--dataset_config`) that points to raw media + captions and a cache directory.
The caching scripts populate that cache directory with `.safetensors` files which are then merged into the training batch.

### Typical raw dataset layout

The exact dataset format depends on your dataset config, but a common layout is:

```
dataset_root/
  videos/
    000001.mp4
    000001.txt
    000002.mp4
    000002.txt
  audio/                    # optional (T2AV)
    000001.wav
    000002.wav
```

Notes:

- Captions are taken from the dataset config (often sidecar `.txt` files).
- Audio caching resolves audio files by default using the same basename as the video/image item key. You can override this using `--audio_dir` and `--audio_ext` when running `ltxv2_cache_audio_latents.py`.

### Typical cache directory layout

The dataset config usually specifies a `cache_directory`. After caching, you should expect files like:

```
cache_directory/
  000001_1024x0576_ltxv2.safetensors          # video latents
  000001_ltxv2_te.safetensors                 # text encoder outputs
  000001_ltxv2_audio.safetensors              # audio latents (AV only)
```

The dataset loader merges the latent cache, optional audio sidecar, and text cache into one training batch.

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

### Example: video-only (T2V)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltxv2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode video \
  --ltxv2_model /path/to/ltxv2_dit.safetensors \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --ltxv2_first_frame_conditioning_p 0.1 \
  --network_module musubi_tuner.networks.lora_ltxv2 \
  --network_dim 16 \
  --network_alpha 16
```

### Example: audio-video (AV)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltxv2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode av \
  --ltxv2_model /path/to/ltxav_dit.safetensors \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --ltxv2_first_frame_conditioning_p 0.1 \
  --audio_loss_weight 1.0 \
  --network_module musubi_tuner.networks.lora_ltxv2 \
  --network_dim 16 \
  --network_alpha 16
```

## End-to-end example workflow

### Step 1: cache video latents

```bash
python ltxv2_cache_latents.py \
  --dataset_config /path/to/dataset.toml \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --vae_dtype bf16
```

### Step 2: cache text encoder outputs

Video-only (T2V):

```bash
python ltxv2_cache_text_encoder_outputs.py \
  --dataset_config /path/to/dataset.toml \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --mixed_precision bf16
```

Audio-video (T2AV): cache concatenated `[video_ctx, audio_ctx]` embeddings:

```bash
python ltxv2_cache_text_encoder_outputs.py \
  --dataset_config /path/to/dataset.toml \
  --ltxv2_audio_video \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --mixed_precision bf16
```

### Step 3: cache audio latents (T2AV only)

```bash
python -m musubi_tuner.ltxv2_cache_audio_latents \
  --dataset_config /path/to/dataset.toml \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --audio_dir /path/to/dataset_root/audio \
  --audio_ext .wav \
  --audio_dtype fp16
```

### Step 4: train

Use the training commands above (T2V or T2AV).

## Preview sampling (`--sample_prompts`)

If `--sample_prompts` is provided, the trainer will:

- Load the official Gemma text encoder using `--ltx2_checkpoint` + `--gemma_root`
- Encode prompt(s) to hidden states
- Run a FlowMatchDiscreteScheduler denoising loop
- Decode latents with the VAE and save preview output

## Loss weighting

LTXV2 supports the following loss weights:

- `--video_loss_weight`
- `--audio_loss_weight` (AV only)

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
