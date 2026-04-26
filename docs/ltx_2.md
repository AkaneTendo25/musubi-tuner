# LTX-2 / LTX-2.3

Supports LoRA training for both **LTX-2 (19B)** and **LTX-2.3 (22B)** models with the following training modes: text-to-video, joint audio-video, audio-only, IC-LoRA / video-to-video, and audio-reference IC-LoRA.

### Supported Model Versions

| Version | Parameters | Key Differences |
|---------|-----------|-----------------|
| LTX-2 (19B) | 19B | Single `aggregate_embed`, caption projection inside transformer |
| LTX-2.3 (22B) | 22B | Dual `video_aggregate_embed`/`audio_aggregate_embed`, caption projection moved to feature extractor (`caption_proj_before_connector`), cross-attention AdaLN (`prompt_adaln`), separate audio connector dimensions, BigVGAN v2 vocoder with bandwidth extension |

Version choice for training is controlled by `--ltx_version` (default: `2.3`) in `ltx2_train_network.py`. The trainer auto-detects the checkpoint version from metadata and warns on mismatch.

Caching scripts (`ltx2_cache_latents.py`, `ltx2_cache_text_encoder_outputs.py`) do not use `--ltx_version`; they work with both LTX-2 and LTX-2.3 checkpoints directly via `--ltx2_checkpoint`.

---

## Table of Contents

- [Installation](#installation)
  - [CUDA Version](#cuda-version)
  - [Downloading Required Models](#downloading-required-models)
- [Supported Dataset Types](#supported-dataset-types)
- [1. Caching Latents](#1-caching-latents)
  - [Latent Caching Command](#latent-caching-command)
  - [Latent Caching Arguments](#latent-caching-arguments)
  - [Latent Cache Output Files](#latent-cache-output-files)
  - [Memory Optimization for Caching](#memory-optimization-for-caching)
- [2. Caching Text Encoder Outputs](#2-caching-text-encoder-outputs)
  - [Text Encoder Caching Arguments](#text-encoder-caching-arguments)
  - [Text Encoder Output Files](#text-encoder-output-files)
  - [Loading Gemma from a Single Safetensors File](#loading-gemma-from-a-single-safetensors-file)
- [3. Training](#3-training)
  - [Choosing Model Version for Training (2.0 vs 2.3)](#choosing-model-version-for-training-20-vs-23)
  - [Source-Free Training from Cache](#optional-source-free-training-from-cache)
  - [Standard LoRA Training](#standard-lora-training)
  - [Advanced: LyCORIS/LoKR Training](#advanced-lycorislokr-training)
  - [Training Arguments](#training-arguments)
    - [Memory Optimization](#memory-optimization)
      - [Quantization Options](#quantization-options)
      - [Other Memory Options](#other-memory-options)
    - [Aggressive VRAM Optimization (8-16GB GPUs)](#aggressive-vram-optimization-8-16gb-gpus)
    - [NF4 Quantization](#nf4-quantization)
    - [Model Version](#model-version)
    - [Audio-Video Support](#audio-video-support)
    - [Loss Function Type](#loss-function-type)
    - [Loss Weighting](#loss-weighting)
    - [Additional Audio Training Flags](#additional-audio-training-flags)
    - [Modality Freezing (G2D)](#modality-freezing-g2d)
    - [Per-Module Learning Rates](#per-module-learning-rates)
    - [Per-Module LoRA Rank](#per-module-lora-rank)
    - [Adaptive LoRA Rank](#adaptive-lora-rank)
    - [Preservation & Regularization](#preservation--regularization)
    - [Self-Flow (Self-Supervised Flow Matching)](#self-flow-self-supervised-flow-matching)
    - [HFATO (High-Frequency Awareness Training Objective)](#hfato-high-frequency-awareness-training-objective)
    - [Timestep Sampling](#timestep-sampling)
    - [LoRA Targets](#lora-targets)
      - [LoRA Target Estimation](#lora-target-estimation-ltx2_estimatepy)
      - [Connector LoRA](#connector-lora---train_connectors)
    - [IC-LoRA / Video-to-Video Training](#ic-lora--video-to-video-training)
    - [Audio-Reference IC-LoRA](#audio-reference-ic-lora)
    - [Sampling with Tiled VAE](#sampling-with-tiled-vae)
    - [Precached Sample Prompts](#precached-sample-prompts)
    - [Two-Stage Sampling (WIP)](#two-stage-sampling-wip)
    - [Checkpoint Output Format](#checkpoint-output-format)
    - [Resuming Training](#resuming-training)
- [Merge LoRA into Base Model](#merge-lora-into-base-model)
- [Merge LTX-2 LoRAs](#merge-ltx-2-loras)
  - [LoRA Merge Arguments](#lora-merge-arguments)
- [Dataset Configuration](#dataset-configuration)
  - [Video Dataset Options](#video-dataset-options)
  - [Audio Dataset Options](#audio-dataset-options)
  - [Example TOML](#example-toml)
  - [Frame Rate (FPS) Handling](#frame-rate-fps-handling)
- [Validation Datasets](#validation-datasets)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
  - [Audio/Voice Training with Mixed Datasets](#audiovoice-training-with-mixed-datasets)
  - [Technical Notes](#technical-notes)
- [4. Slider LoRA Training](#4-slider-lora-training)
  - [4a. Text-Only Mode](#4a-text-only-mode)
  - [4b. Reference Mode](#4b-reference-mode)
  - [Slider Tips](#slider-tips)
- [Setup / Update Script](#setup--update-script)
- [References](#references)

---

## Installation

The base installation procedure is the same as upstream musubi-tuner — follow the [Installation guide](../README.md#installation) (`pip install -e .` in a virtual environment). The sections below cover LTX-2-specific requirements (CUDA version, model downloads) that go on top of the base install.

Unless otherwise noted, command examples in this LTX-2 guide were tested on Windows 11. They should also work on Linux, but you may need small shell/path adjustments.

For a Windows-focused community setup example for this fork (tested environment and install helpers), see [Discussion #19: Windows OS installation/usage helpers](https://github.com/AkaneTendo25/musubi-tuner/discussions/19).

### CUDA Version

The PyTorch install command must use a CUDA version compatible with your GPU. Adjust the `--index-url` accordingly:

```bash
# Default (most GPUs, including RTX 30xx/40xx):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# RTX 5090 / 50xx series (Blackwell):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Always match the CUDA version to your GPU architecture — check [PyTorch's compatibility matrix](https://pytorch.org/get-started/locally/) for the latest supported versions.

### Downloading Required Models

> [!WARNING]
> The dashboard UI and the Windows Setup / Update script are still early prototypes and work in progress. Their stable behavior is not guaranteed yet, and some flows may still break or change.

You can now handle the common model downloads directly from the dashboard:

- Use the project page to choose a template that matches your use case.
- Open `Caching`, `Training`, or `Inference` and use the download actions beside the LTX-2 and Gemma model fields.
- Use `Setup & Updates` in the dashboard to verify the install, repo state, shortcuts, and project readiness before you start caching or training.

Manual download is still fully supported, and it remains useful if you want to manage checkpoints outside the default model directory.

**LTX-2 Checkpoint** — use as `--ltx2_checkpoint`:
- LTX-2 (19B): [ltx-2-19b-dev.safetensors](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors)
- LTX-2.3 (22B): [ltx-2.3-22b-dev.safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)

**Gemma Text Encoder** — pick one:
- HF directory (`--gemma_root`): [gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
- Single file (`--gemma_safetensors`): [gemma_3_12B_it_fp8_e4m3fn.safetensors](https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors)

Other Gemma 3 12B variants may work but not all have been tested.

---

## Supported Dataset Types

| Mode | Dataset Type | Notes |
|------|--------------|-------|
| `video` | Images | Treated as 1-frame samples (`F=1`) |
| `video` | Videos | Standard video training |
| `av` | Videos with audio | Audio extracted from video or external audio files |
| `audio` | Audio only | Dataset must be audio-only; training uses audio-driven latent geometry |

---

## 1. Caching Latents

This step pre-processes media files into VAE latents to speed up training.

**Script:** `ltx2_cache_latents.py`

### Latent Caching Command
```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --device cuda ^
  --vae_dtype bf16 ^
  --ltx2_mode av ^
  --ltx2_audio_source video
```

### Latent Caching Arguments
- `--ltx2_mode`, `--ltx_mode`: Caching modality selector. Default is video-only (`v`/`video`). Use `av` to cache both `*_ltx2.safetensors` (video) and `*_ltx2_audio.safetensors` (audio) latents.
- `--ltx2_audio_source video|audio_files`: Use audio from the video or from external files.
- `--ltx2_audio_dir`, `--ltx2_audio_ext`: Optional when using `--ltx2_audio_source audio_files` (default extension: `.wav`).
- `--ltx2_checkpoint`: Required for `--ltx2_mode av` or `--ltx2_mode audio`.
- `--audio_only_target_resolution`: Optional square override for audio-only latent geometry. Only takes effect when `--audio_only_sequence_resolution 0`; otherwise the fixed sequence resolution is used instead.
- `--audio_only_target_fps`: Target FPS used to derive audio-only frame counts from audio duration (default: `25`).
- `--audio_video_latent_channels`: Optional override for audio-only video latent channels (auto-detected from checkpoint by default).
- `--ltx2_audio_dtype`: Data type for audio VAE encoding (default: `float16`).
- `--audio_video_latent_dtype`: Optional override for audio-only video latent dtype (defaults to `--ltx2_audio_dtype`).
- `--vae_dtype`: Data type for VAE latents (default comes from the cache script).
- `--save_dataset_manifest`: Optional. Saves a cache-only dataset manifest for source-free training.
- `--precache_sample_latents`: Cache I2V conditioning image latents for sample prompts, then continue with normal latent caching. Requires `--sample_prompts`.
- `--sample_latents_cache`: Path for the I2V conditioning latents cache file (default: `<cache_dir>/ltx2_sample_latents_cache.pt`).
- `--reference_frames`: Number of reference frames to cache for IC-LoRA / V2V (default: `1`).
- `--reference_downscale`: Spatial downscale factor for cached reference latents (default: `1`).

### Latent Cache Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2.safetensors` | Video latents: `latents_{F}x{H}x{W}_{dtype}`. In audio-only mode, this file also stores `ltx2_virtual_num_frames_int32` (used for timestep sampling) and `ltx2_virtual_height_int32`/`ltx2_virtual_width_int32` (only used when `--audio_only_sequence_resolution 0`). |
| `*_ltx2_audio.safetensors` | Audio latents: `audio_latents_{T}x{mel_bins}x{channels}_{dtype}`, `audio_lengths_int32` |

### Memory Optimization for Caching
If you encounter Out-Of-Memory (OOM) errors during caching (especially with higher resolutions like 1080p), you have two options:

**Option 1: VAE temporal chunking** (fewer parameters, for moderate OOM)
```bash
python ltx2_cache_latents.py ^
  ...
  --vae_chunk_size 16
```
- `--vae_chunk_size`: Processes video in temporal chunks (e.g., 16 or 32 frames at a time). Default: `None` (all frames).

**Option 2: VAE tiled encoding** (larger VRAM savings, for severe OOM or high-resolution videos)
```bash
python ltx2_cache_latents.py ^
  ...
  --vae_spatial_tile_size 512 ^
  --vae_spatial_tile_overlap 64
```
- `--vae_spatial_tile_size`: Splits each frame into spatial tiles of this size in pixels (e.g., 512). Must be >= 64 and divisible by 32. Default: `None` (disabled).
- `--vae_spatial_tile_overlap`: Overlap between spatial tiles in pixels. Must be divisible by 32. Default: `64`.
- `--vae_temporal_tile_size`: Splits the video into temporal tiles of this many frames (e.g., 64). Must be >= 16 and divisible by 8. Default: `None` (disabled).
- `--vae_temporal_tile_overlap`: Overlap between temporal tiles in frames. Must be divisible by 8. Default: `24`.

Spatial and temporal tiling can be combined. Tiled encoding trades speed for VRAM savings.

Both options can be combined (e.g., `--vae_chunk_size 16 --vae_spatial_tile_size 512`).

---

## 2. Caching Text Encoder Outputs

This step pre-computes text embeddings using the Gemma text encoder.

**Script:** `ltx2_cache_text_encoder_outputs.py`

### Text Encoder Caching Command
```bash
python ltx2_cache_text_encoder_outputs.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --gemma_root /path/to/gemma ^
  --gemma_load_in_8bit ^
  --device cuda ^
  --mixed_precision bf16 ^
  --ltx2_mode av ^
  --batch_size 1
```

### Text Encoder Caching Arguments
- `--gemma_root`: Path to the local Gemma model folder (HuggingFace format). Required unless `--gemma_safetensors` is used.
- `--gemma_safetensors`: Path to a single Gemma `.safetensors` file (e.g. an FP8 export from ComfyUI). Loads weights, config, and tokenizer from one file — no `--gemma_root` needed. See [Loading Gemma from a Single Safetensors File](#loading-gemma-from-a-single-safetensors-file) below.
- `--gemma_load_in_8bit`: Loads Gemma in 8-bit quantization. Cannot be combined with `--gemma_safetensors`.
- `--gemma_load_in_4bit`: Loads Gemma in 4-bit quantization. Cannot be combined with `--gemma_safetensors`.
- `--gemma_bnb_4bit_quant_type nf4|fp4`: Quantization type for 4-bit loading (default: `nf4`).
- `--gemma_bnb_4bit_disable_double_quant`: Disable bitsandbytes double quantization for 4-bit loading.
- `--gemma_bnb_4bit_compute_dtype auto|fp16|bf16|fp32`: Compute dtype for 4-bit operations (default: `auto`, uses `--mixed_precision` dtype).
- `--ltx2_checkpoint`: Required. Use `--ltx2_text_encoder_checkpoint` to override for text encoder connector weights.
- `--cache_before_connector`: Also save pre-connector text features (`video_features_{dtype}`, `audio_features_{dtype}`) alongside standard post-connector embeddings. Required for `--train_connectors` during training. Does not change standard cache keys; only adds extra tensors.
- 8-bit/4-bit loading requires `--device cuda`.

> [!IMPORTANT]
> `--ltx2_mode` / `--ltx_mode` **must match** the mode used during latent caching. Default is `video`; use `av` to concatenate video and audio prompt embeddings.

### Text Encoder Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2_te.safetensors` | `video_prompt_embeds_{dtype}`, `audio_prompt_embeds_{dtype}` (av only), `prompt_attention_mask`, `text_{dtype}`, `text_mask` |
| (with `--cache_before_connector`) | Above keys plus `video_features_{dtype}`, `audio_features_{dtype}` (av only) |

### Loading Gemma from a Single Safetensors File

`--gemma_safetensors` loads Gemma from a single `.safetensors` file instead of a HuggingFace model directory. Weights, tokenizer (`spiece_model` key), and config (inferred from tensor shapes) are all read from the one file. No `--gemma_root` needed.

```bash
python ltx2_cache_text_encoder_outputs.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --gemma_safetensors /path/to/gemma3-12b-it-fp8.safetensors ^
  --device cuda ^
  --mixed_precision bf16
```

- FP8 weights (`F8_E4M3` / `F8_E5M2`) are detected automatically and kept in FP8 on GPU (compute in bf16).
- `--gemma_fp8_weight_offload` / `--no-gemma_fp8_weight_offload`: Explicitly enable or disable CPU offload for FP8 Gemma linear weights when using `--gemma_safetensors`.
- If `--gemma_fp8_weight_offload` is omitted, the code falls back to `LTX2_GEMMA_SAFETENSORS_WEIGHT_OFFLOAD` (default environment fallback: enabled / `1`).
- `--gemma_load_in_8bit` / `--gemma_load_in_4bit` cannot be combined with `--gemma_safetensors`.
- If the file has no `spiece_model` key, tokenizer extraction fails — use `--gemma_root` instead.
- Works in all scripts that load Gemma: `ltx2_cache_text_encoder_outputs.py`, `ltx2_train_network.py`, `ltx2_train_slider.py`, `ltx2_generate_video.py`.

---

## 3. Training

Launch the training loop using `accelerate`.

**Script:** `ltx2_train_network.py`

### Choosing Model Version for Training (2.0 vs 2.3)

Use this rule:

| Checkpoint you train on | Required training flags |
|---|---|
| LTX-2 (19B) checkpoint | `--ltx_version 2.0` |
| LTX-2.3 (22B) checkpoint | `--ltx_version 2.3` |

Recommended practice:
- Always set `--ltx_version` explicitly in training commands (do not rely on the default).
- On first run, set `--ltx_version_check_mode error` to fail fast if the selected version does not match checkpoint metadata.
- After validation, you can switch to `--ltx_version_check_mode warn`.
- Pre-quantized FP8 checkpoints (e.g. `ltx-2.3-22b-dev-fp8.safetensors`) are supported. The loader auto-detects `weight_scale`/`input_scale` keys and dequantizes to bf16 before applying LoRA merges and any further quantization.

When changing checkpoints (important):
- If you change `--ltx2_checkpoint` (e.g., LTX-2 -> LTX-2.3, or different 2.3 variant), re-run **both** caches:
  - `ltx2_cache_latents.py`
  - `ltx2_cache_text_encoder_outputs.py`
- Do not reuse old `*_ltx2_te.safetensors` from a different checkpoint. For LTX-2.3 audio/av training this can cause context/mask shape mismatches (for example FlashAttention varlen mask-length errors).
- If you use `--dataset_manifest`, regenerate it from the recache step so training points to the new cache files.
- Pre-quantized FP8 checkpoints (`*fp8*.safetensors`) work with both `--fp8_base` and `--fp8_base --fp8_scaled`. The loader dequantizes to bf16 using the checkpoint's scale tensors before any further processing.

Example (LTX-2.3 training):
```bash
--ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
--ltx_version 2.3 ^
--ltx_version_check_mode error
```

### Optional: Source-Free Training from Cache
If you cached with `--save_dataset_manifest`, you can train without source dataset paths:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --dataset_manifest dataset_manifest.json ^
  ... (other training args)
```

Use `--dataset_manifest` instead of `--dataset_config`.

### Standard LoRA Training
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --gemma_load_in_8bit ^
  --gemma_root /path/to/gemma ^
  --separate_audio_buckets ^
  --ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
  --ltx_version 2.3 ^
  --ltx_version_check_mode error ^
  --ltx2_mode av ^
  --fp8_base ^
  --fp8_scaled ^
  --blocks_to_swap 10 ^
  --sdpa ^
  --gradient_checkpointing ^
  --learning_rate 1e-4 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 ^
  --network_alpha 32 ^
  --timestep_sampling shifted_logit_normal ^
  --sample_at_first ^
  --sample_every_n_epochs 5 ^
  --sample_prompts sampling_prompts.txt ^
  --sample_with_offloading ^
  --sample_tiled_vae ^
  --sample_vae_tile_size 512 ^
  --sample_vae_tile_overlap 64 ^
  --sample_vae_temporal_tile_size 48 ^
  --sample_vae_temporal_tile_overlap 8 ^
  --sample_merge_audio ^
  --output_dir output ^
  --output_name ltx23_lora
```

Pre-quantized FP8 checkpoints (`*fp8*.safetensors`) are supported — `--fp8_base --fp8_scaled` works the same as with standard checkpoints (weights are dequantized to bf16 first, then re-quantized).

For LTX-2 checkpoints, replace:
- `--ltx2_checkpoint /path/to/ltx-2.3.safetensors` -> `--ltx2_checkpoint /path/to/ltx-2.safetensors`
- `--ltx_version 2.3` -> `--ltx_version 2.0`

### Advanced: LyCORIS/LoKR Training

musubi-tuner supports advanced LoRA algorithms (LoKR, LoHA, LoCoN, etc.) via:
- `--network_args` for inline `key=value` settings
- `--lycoris_config <path.toml>` for TOML-based settings

See the [LyCORIS algorithm list](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md) and [guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md) for algorithm details and recommended settings. You can also refer to the local [LoHa/LoKr documentation](./loha_lokr.md).

No bundled example TOML files are shipped; provide your own config path.

```bash
# Install LyCORIS first
pip install lycoris-lora

# Example TOML (save anywhere, e.g. my_lycoris.toml)
# [network]
# base_algo = "lokr"
# base_factor = 16
#
# [network.modules."*audio*"]
# algo = "lora"
# dim = 64
# alpha = 32
#
# [network.init]
# lokr_norm = 1e-3

# LoKR example
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as above) ^
  --lora_target_preset lycoris ^
  --network_module lycoris.kohya ^
  --lycoris_config my_lycoris.toml ^
  --output_name ltx2_lokr

# LoCoN example (inline args)
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as above) ^
  --network_module lycoris.kohya ^
  --network_args "algo=locon" "conv_dim=8" "conv_alpha=4" ^
  --output_name ltx2_locon
```

**Target format (`--network_args`)**
- Pass args as space-separated `key=value` pairs.
- Each pair is one argument token: `--network_args "key1=value1" "key2=value2"`.
- Common LyCORIS keys: `algo`, `factor`, `conv_dim`, `conv_alpha`, `dropout`.
- Use `--init_lokr_norm` only with LoKR (`algo=lokr`).
- If both TOML and `--network_args` are used, `--network_args` can override nested TOML keys with:
  - `modules.<name>.<param>=...`
  - `init.<param>=...`

`--lycoris_config` requires `--network_module lycoris.kohya`.

### Training Arguments

All training arguments can be placed in a `.toml` config file instead of on the command line via `--config_file config.toml`. See the [configuration files guide](./advanced_config.md#using-configuration-files-to-specify-training-options) for format details.

#### Memory Optimization

For additional training and inference speedups, see the [torch.compile Support](./torch_compile.md) documentation.

##### Quantization Options

| Method | VRAM (19B, LTX-2) | Weight Error (MAE) | SNR | Cosine Similarity |
|--------|------------------|--------------------|-----|-------------------|
| BF16 (baseline) | ~38 GB | 0.0011 | 55.6 dB | 0.999999 |
| `--fp8_base --fp8_scaled` | ~19 GB | 0.0171 (15x BF16) | 32.0 dB | 0.999686 |
| `--nf4_base` | ~10 GB | 0.0678 (60x BF16) | 21.2 dB | 0.996188 |
| `--nf4_base --loftq_init` | ~10 GB | 0.0654 (60x BF16) | 21.5 dB | 0.996437 |

*Approximate values measured on random N(0,1) weights with shapes representative of LTX-2 transformer layers. MAE = mean absolute error between original and dequantized weights. LoftQ error is measured after adding the LoRA correction (rank 32, 2 iterations). No benchmark script is included in this repo.*

NF4 has ~4x higher weight error than FP8 (cosine 0.996 vs 0.9997). The base model is frozen during LoRA training, so the quantization error is constant rather than accumulating. LoftQ initializes LoRA weights from the quantization residual via SVD.

- `--fp8_base`: keep base model weights in FP8 path (~19 GB VRAM).
- `--fp8_scaled`: quantize checkpoint weights to FP8 at load time. Works with both standard (bf16/fp16/fp32) and pre-quantized FP8 checkpoints (the latter are dequantized to bf16 first, then re-quantized).
- `--nf4_base`: NF4 4-bit quantization (~10 GB VRAM). Mutually exclusive with `--fp8_base`. See [NF4 Quantization](#nf4-quantization) below.
- `--quantize_device cpu|cuda|gpu`: Device for NF4/FP8 quantization at startup (default: `cuda`). `cpu` loads and quantizes weights on CPU, then moves to GPU. `cuda` loads and quantizes directly on GPU. Overrides `LTX2_NF4_CALC_DEVICE` / `LTX2_FP8_CALC_DEVICE` env vars.

##### Other Memory Options

| Argument | Description |
|----------|-------------|
| `--blocks_to_swap X` | Offload X transformer blocks to CPU (max 47 for 48-block model). Higher = more VRAM saved, more CPU↔GPU overhead |
| `--use_pinned_memory_for_block_swap` | Use pinned memory for faster CPU↔GPU block transfers |
| `--gradient_checkpointing` | Reduce VRAM by recomputing activations during backward pass |
| `--gradient_checkpointing_cpu_offload` | Offload activations to CPU during gradient checkpointing |
| `--ffn_chunk_target` | `all`, `video`, or `audio` — enable FFN chunking for selected modules |
| `--ffn_chunk_size N` | Chunk size for FFN chunking (0 = disabled) |
| `--split_attn_target` | `none`, `all`, `self`, `cross`, `text_cross`, `av_cross`, `video`, `audio` — split attention target modules |
| `--split_attn_mode` | `batch` or `query` — split by batch dimension or query length |
| `--split_attn_chunk_size N` | Chunk size for query-based split attention (0 = default 1024) |
| `--sdpa` | Use PyTorch scaled dot-product attention (recommended default) |
| `--flash_attn` | Use FlashAttention 2 (requires `flash-attn` package built for your CUDA + PyTorch) |
| `--flash3` | Use FlashAttention 3 (requires `flash-attn` v3 with Hopper+ GPU) |

#### Aggressive VRAM Optimization (8-16GB GPUs)

For maximum VRAM savings on 8-16GB GPUs, use this combination of flags. See also the [Advanced Configuration guide](./advanced_config.md) for optimizer options (`--optimizer_type`, `--lr_scheduler`, Schedule-Free optimizer, etc.):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --ltx2_mode av ^
  --gemma_load_in_8bit ^
  --gemma_root /path/to/gemma ^
  --fp8_base ^
  --fp8_scaled ^
  --blocks_to_swap 47 ^
  --use_pinned_memory_for_block_swap ^
  --gradient_checkpointing ^
  --gradient_checkpointing_cpu_offload ^
  --sdpa ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 ^
  --network_alpha 16 ^
  --sample_with_offloading ^
  --sample_tiled_vae ^
  --sample_vae_tile_size 512 ^
  --sample_vae_temporal_tile_size 48 ^
  --output_dir output
```

**Tips for low-VRAM training:**
- Use `--fp8_base --fp8_scaled` (works with both standard and pre-quantized FP8 checkpoints)
- Use `--blocks_to_swap 47` (keeps only 1 block on GPU)
- Use smaller LoRA rank (`--network_dim 16` instead of 32)
- Use smaller training resolutions (e.g., 512x320)
- Reduce `--sample_vae_temporal_tile_size` to 24 or lower
- Use `--use_pinned_memory_for_block_swap` - faster transfers

#### NF4 Quantization

NF4 (4-bit NormalFloat) quantization uses a 16-value codebook optimized for normally-distributed weights (QLoRA paper). Weights are stored as packed uint8 with per-block absmax scaling. VRAM usage is ~10 GB vs ~19 GB for FP8 and ~38 GB for BF16.

**Basic usage:**
```bash
accelerate launch ... ltx2_train_network.py ^
  --nf4_base ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 ^
  ...
```

**With LoftQ initialization:**

LoftQ pre-computes LoRA A/B matrices from the truncated SVD of the NF4 quantization residual (`W - dequant(Q(W))`). This runs once at startup and adds no runtime cost.

```bash
accelerate launch ... ltx2_train_network.py ^
  --nf4_base ^
  --loftq_init ^
  --loftq_iters 2 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 ^
  ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--nf4_base` | off | Enable NF4 4-bit quantization for the base model |
| `--nf4_block_size` | 32 | Elements per quantization block |
| `--loftq_init` | off | LoftQ initialization for LoRA (requires `--nf4_base`) |
| `--loftq_iters` | 2 | Number of alternating quantize-SVD iterations |
| `--awq_calibration` | off | Experimental: activation-aware channel scaling before quantization |
| `--awq_alpha` | 0.25 | AWQ scaling strength (0 = no effect, 1 = full) |
| `--awq_num_batches` | 8 | Number of synthetic calibration batches for AWQ |
| `--quantize_device` | `cuda` | Device for quantization math (`cpu`, `cuda`, `gpu`) |

**Pre-quantized models (recommended):**

By default, NF4 quantization runs from scratch on every startup. `ltx2_quantize_model.py` quantizes once and saves the result (~42% of the original file size). The training/inference code auto-detects pre-quantized checkpoints via safetensors metadata and skips re-quantization.

```bash
python src/musubi_tuner/ltx2_quantize_model.py ^
  --input_model path/to/ltx-2.3-22b-dev.safetensors ^
  --output_model path/to/ltx-2.3-22b-dev-nf4.safetensors ^
  --loftq_init --network_dim 32
```

Output files (kept in the same directory):
- `*-nf4.safetensors` — quantized model (transformer in NF4, VAE and other components unchanged)
- `*-nf4.loftq_r32.safetensors` — pre-computed LoftQ init for rank 32 (only with `--loftq_init`)

Then use it exactly like the original checkpoint — just point `--ltx2_checkpoint` at the NF4 file. `--nf4_base` is still required (enables the runtime dequantization patch):

```bash
accelerate launch ... ltx2_train_network.py ^
  --ltx2_checkpoint path/to/ltx-2.3-22b-dev-nf4.safetensors ^
  --nf4_base --loftq_init --network_dim 32 ^
  ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_model` | required | Path to original `.safetensors` checkpoint |
| `--output_model` | required | Path for quantized output |
| `--nf4_block_size` | 32 | Elements per quantization block |
| `--calc_device` | `cuda` if available | Device for quantization computation |
| `--loftq_init` | off | Pre-compute LoftQ initialization (requires `--network_dim`) |
| `--loftq_iters` | 2 | Number of LoftQ alternating iterations |
| `--network_dim` | 0 | LoRA rank for LoftQ (must match training `--network_dim`) |

- LoftQ is rank-specific: changing `--network_dim` requires re-running the quantize script with the new rank. The quantized model itself does not need to be regenerated.
- `--awq_calibration` is incompatible with pre-quantized models (requires full-precision weights).
- The quantized output is bit-for-bit identical to dynamic quantization on the same device.

**Notes:**
- `--nf4_base` and `--fp8_base` are mutually exclusive.
- `--loftq_init` requires `--nf4_base`.
- `--awq_calibration` is experimental. Adds a per-layer division during forward passes. In synthetic tests, reduces activation-weighted error by ~3-5%; effect on real training quality has not been validated.
- Compatible with `--blocks_to_swap`, `--gradient_checkpointing`, and other training options. NF4 reduces block swap transfer size (4-bit vs 16-bit per weight).
- Quantization targets transformer block weights only. Embedding layers, norms, and projection layers remain in full precision.

#### Model Version
- `--ltx_version 2.0|2.3`: Select target model version (default: `2.3`). Controls default behavior for version-dependent settings (e.g., `--shifted_logit_mode` defaults to `legacy` for 2.0, `stretched` for 2.3).
- `--ltx_version_check_mode off|warn|error`: How to handle mismatch between `--ltx_version` and checkpoint metadata (default: `warn`). The trainer reads checkpoint config keys (`cross_attention_adaln`, `caption_proj_before_connector`, `bwe` vocoder) to detect the actual version.

#### Audio-Video Support
- `--ltx2_mode`, `--ltx_mode`: Training modality selector. Default is `v` (`video`). Values: `video`, `av`, `audio` (aliases: `v`, `va`, `a`).
- `--ltx2_audio_only_model`: Force loading a physically audio-only transformer variant (video modules omitted). Requires `--ltx2_mode audio`.
- `--separate_audio_buckets`: Keeps audio and non-audio items in separate batches (reduces VRAM for image/video-only batches).
- `--audio_bucket_strategy pad|truncate`: Audio duration bucketing strategy. `pad` (default) rounds to nearest bucket boundary and pads shorter clips with loss masking. `truncate` floors to bucket boundary and truncates all clips to bucket length (no padding or masking needed).
- `--audio_bucket_interval`: Audio bucket step size in seconds (default: `2.0`). Controls how finely audio clips are grouped by duration.
- `--min_audio_batches_per_accum`: Minimum number of audio-bearing microbatches per gradient accumulation window.
- `--audio_batch_probability`: Probability of selecting an audio-bearing batch when both audio and non-audio batches are available.
  - `--min_audio_batches_per_accum` and `--audio_batch_probability` are mutually exclusive.
- `--caption_dropout_rate`: Probability of dropping ALL text conditioning (video + audio) for a sample during training (default: `0.0`, disabled). When triggered, the sample's text embeddings are zeroed out and the attention mask is cleared, training the model to generate without text guidance. This enables classifier-free guidance (CFG) at inference.
- `--video_caption_dropout_rate`: Probability of dropping only the video text conditioning while keeping audio text conditioning (default: `0.0`). AV mode only. Applied independently per sample before `--caption_dropout_rate`.
- `--audio_caption_dropout_rate`: Probability of dropping only the audio text conditioning while keeping video text conditioning (default: `0.0`). AV mode only. Applied independently per sample before `--caption_dropout_rate`.

The three dropout rates are independent. For a given sample, the per-modality dropout is applied first (on the separate `video_prompt_embeds` / `audio_prompt_embeds`), then the joint dropout is applied on the concatenated result. A sample can end up with: both modalities present, video-only, audio-only, or fully unconditional.

#### Loss Function Type

`--loss_type` selects the element-wise loss function used for both video and audio branches. Default is `mse`.

| `--loss_type` | PyTorch function | Per-element formula |
|---|---|---|
| `mse` (default) | `F.mse_loss` | `(pred - tgt)²` |
| `mae` / `l1` | `F.l1_loss` | `\|pred - tgt\|` |
| `huber` / `smooth_l1` | `F.smooth_l1_loss` | `0.5·(pred-tgt)²/δ` when `\|pred-tgt\| < δ`, else `\|pred-tgt\| - 0.5·δ` |

- `--huber_delta` (float, default: 1.0): Transition point for Huber loss. Only used when `--loss_type` is `huber` or `smooth_l1`. Smaller values make the loss behave more like L1; larger values more like MSE.

All other training mechanics (weighting scheme, masking, audio balancing) apply on top of the chosen loss unchanged.

```bash
# L1 loss
--loss_type mae

# Huber with tighter quadratic region
--loss_type huber --huber_delta 0.1
```

#### Loss Weighting
- `--video_loss_weight`: Weight for video loss (default: 1.0).
- `--audio_loss_weight`: Weight for audio loss in AV mode (default: 1.0).
- Dataset config `video_loss_weight` / `audio_loss_weight` override the corresponding CLI weight for that dataset only.
- `--audio_loss_balance_mode`: Audio loss balancing strategy. Values: `none` (default), `inv_freq`, `ema_mag`, `uncertainty`, `ogm_ge`.
- `--audio_loss_balance_min`, `--audio_loss_balance_max`: Clamp range for effective audio weight (defaults: 0.05, 4.0). Used by `inv_freq` and `ema_mag` only.

**`inv_freq` mode** — inverse-frequency reweighting for mixed audio/non-audio training. Boosts audio loss proportionally to how rare audio batches are.

- `--audio_loss_balance_beta`: EMA update rate for observed audio-batch frequency (default: 0.01).
- `--audio_loss_balance_eps`: Denominator floor for inverse-frequency scaling (default: 0.05).
- `--audio_loss_balance_ema_init`: Initial audio-frequency EMA value (default: 1.0).

Example:
```bash
--audio_loss_weight 1.0 ^
--audio_loss_balance_mode inv_freq ^
--audio_loss_balance_beta 0.01 ^
--audio_loss_balance_eps 0.05 ^
--audio_loss_balance_min 0.05 ^
--audio_loss_balance_max 3.0
```

Recommended start values:
- `--audio_loss_balance_beta 0.01` (stable EMA, slower reaction; try `0.02-0.05` for faster reaction)
- `--audio_loss_balance_eps 0.05` (safe floor; increase to `0.1` if weights spike too much)
- `--audio_loss_balance_min 0.05 --audio_loss_balance_max 3.0` (conservative clamp range)
- `--audio_loss_balance_ema_init 1.0` (no warm-start boost; use `0.5` only if you want stronger early audio emphasis)

**`ema_mag` mode** — dynamic balancing by matching audio-loss EMA magnitude to a target fraction of video-loss EMA. Bidirectional: can dampen audio (weight < 1.0) when audio loss exceeds `target_ratio * video_loss`, or boost it when audio loss is below that target.

- `--audio_loss_balance_target_ratio`: Target audio/video loss magnitude ratio (default: 0.33 — audio loss targets ~33% of video loss).
- `--audio_loss_balance_ema_decay`: EMA decay for loss magnitude tracking (default: 0.99).

Example:
```bash
--audio_loss_weight 1.0 ^
--audio_loss_balance_mode ema_mag ^
--audio_loss_balance_target_ratio 0.33 ^
--audio_loss_balance_ema_decay 0.99 ^
--audio_loss_balance_min 0.05 ^
--audio_loss_balance_max 4.0
```

Use `ema_mag` when audio and video losses have different natural magnitudes and you want automatic scaling instead of manual `--audio_loss_weight` tuning.

**`uncertainty` mode** — learnable homoscedastic uncertainty weighting (Kendall et al., CVPR 2018). Two log-variance scalars (`log_var_video`, `log_var_audio`) are added to the optimizer and learned jointly with LoRA weights. The combined loss is:

```
loss = 0.5 * exp(-log_var_v) * L_video + 0.5 * log_var_v
     + 0.5 * exp(-log_var_a) * L_audio + 0.5 * log_var_a
```

The regularization terms (`0.5 * log_var`) penalize large variance, preventing either loss from being scaled to zero. Both scalars are initialized to 0.0 and optimized via backpropagation. `--video_loss_weight` and `--audio_loss_weight` are ignored in this mode.

- `--uncertainty_lr`: Learning rate for log-variance parameters (default: same as `--learning_rate`).

Example:
```bash
--audio_loss_balance_mode uncertainty
```

Logged to TensorBoard: `uncertainty/log_var_video`, `uncertainty/log_var_audio`, `uncertainty/precision_video`, `uncertainty/precision_audio`. Higher precision = more weight on that modality's loss. The log-variance params are saved/loaded with checkpoints for training resume.

**`ogm_ge` mode** — Online Gradient Modulation with optional Generalization Enhancement noise. The lower-loss / faster-learning modality is attenuated on each AV step using a discrepancy-dependent coefficient:

```
k = 1 - tanh(alpha * discrepancy)
```

The weaker modality keeps coefficient `1.0`. This is an opt-in conservative implementation for joint AV LoRA training:

- `--ogm_ge_alpha`: Modulation strength (default: `0.3`)
- `--ogm_ge_noise_std`: Optional GE noise scale added to the attenuated modality gradients after backward (default: `0.0`, disabled)

Example:
```bash
--audio_loss_balance_mode ogm_ge ^
--ogm_ge_alpha 0.3 ^
--ogm_ge_noise_std 0.0
```

Logged to TensorBoard: `ogm_ge/video_coeff`, `ogm_ge/audio_coeff`, `ogm_ge/discrepancy`.

On video-only batches (no audio in the current batch), falls back to standard `video_loss * video_weight`.

#### Additional Audio Training Flags

- `--independent_audio_timestep`: Sample a separate timestep for audio (AV/audio modes only).
- `--audio_silence_regularizer`: When AV batches are missing audio latents, use synthetic silence latents instead of skipping the audio branch.
- `--audio_silence_regularizer_weight`: Loss multiplier for synthetic-silence fallback batches.
- `--audio_supervision_mode off|warn|error`: AV audio-supervision monitor mode.
- `--audio_supervision_warmup_steps`: Expected AV batches before supervision checks.
- `--audio_supervision_check_interval`: Run supervision checks every N expected AV batches.
- `--audio_supervision_min_ratio`: Minimum supervised/expected ratio required by the monitor.

#### Modality Freezing (G2D)

Adaptive modality freezing based on per-modality loss EMA ratio (G2D Sequential Modality Prioritization, 2025). When one modality's loss is significantly lower than the other, its LoRA parameters are frozen (`requires_grad=False`) so the under-performing modality can train without gradient interference.

- `--modality_freeze_check_interval <int>`: Check freeze state every N steps. `0` = disabled (default).
- `--modality_freeze_ratio_threshold <float>`: Freeze threshold (default: `0.5`). Audio LoRA is frozen when `audio_loss_ema / video_loss_ema < threshold`. Video LoRA is frozen when the ratio exceeds `1 / threshold`.
- `--modality_freeze_warmup_steps <int>`: Steps before freezing can activate (default: `100`).
- `--modality_freeze_ema_decay <float>`: EMA decay for loss tracking (default: `0.99`).

Example:
```bash
--modality_freeze_check_interval 500 ^
--modality_freeze_ratio_threshold 0.5 ^
--modality_freeze_warmup_steps 200
```

Logged to TensorBoard: `modality_freeze/state` (0=both active, 1=audio frozen, -1=video frozen), `modality_freeze/video_loss_ema`, `modality_freeze/audio_loss_ema`.

#### Per-Module Learning Rates

Set different learning rates for audio vs. video LoRA modules. Useful when audio modules need a lower LR to stabilize AV training.

- `--audio_lr <float>`: Learning rate for all audio LoRA modules (names containing `audio_`). Defaults to `--learning_rate`.
- `--lr_args <pattern=lr> ...`: Regex-based per-module LR overrides. Patterns are matched against LoRA module names via `re.search`.

Priority (highest to lowest): `--lr_args` pattern match > `--audio_lr` catch-all > `--learning_rate` default.

Example:
```bash
--learning_rate 1e-4 ^
--audio_lr 1e-5 ^
--lr_args audio_attn=1e-6 video_to_audio=5e-6
```

Result:
- `audio_attn` modules → 1e-6 (matched by `--lr_args`)
- `video_to_audio` modules → 5e-6 (matched by `--lr_args`)
- Other `audio_*` modules (e.g. `audio_ff`) → 1e-5 (`--audio_lr`)
- Video modules → 1e-4 (`--learning_rate`)

Works with LoRA+ (`loraplus_lr_ratio`): the up/down split applies within each LR group. Both flags default to `None` and are fully backward-compatible. See [LoRA+ in the advanced configuration guide](./advanced_config.md#lora) for setup details.

Optional per-group warmup overrides:

- `--lr_group_warmup_args <pattern=steps> ...`: Override warmup length for matching optimizer groups while keeping the selected scheduler family and default path unchanged. Patterns match group names such as `unet_audio`, `unet_video`, or regex-derived names from `--lr_args`.

Example:
```bash
--learning_rate 1e-4 ^
--audio_lr 3e-5 ^
--lr_scheduler cosine ^
--lr_warmup_steps 500 ^
--lr_group_warmup_args audio=500 video=1500
```

This keeps audio groups on a shorter warmup and video groups on a longer warmup without changing existing behavior unless `--lr_group_warmup_args` is provided.

#### Per-Module LoRA Rank

Set different LoRA rank (dim) for audio vs. video modules.

- `--audio_dim <int>`: LoRA rank for audio modules (names containing `audio_`). Defaults to `--network_dim`.
- `--audio_alpha <float>`: LoRA alpha for audio modules. Defaults to `--network_alpha`.
- `--network_args "cross_modal_dim=<int>"`: Optional LoRA rank override for cross-modal modules (`audio_to_video`, `video_to_audio`, `av_ca_*`).
- `--network_args "cross_modal_alpha=<float>"`: Optional LoRA alpha override for cross-modal modules.

Example:
```bash
--network_dim 32 ^
--network_alpha 16 ^
--audio_dim 8 ^
--audio_alpha 8 ^
--network_args "cross_modal_dim=12" "cross_modal_alpha=12"
```

Result:
- Audio-only modules (`audio_attn`, `audio_ff`, etc.) → rank 8, alpha 8
- Cross-modal modules (`audio_to_video_attn`, `video_to_audio_attn`, `av_ca_*`) → rank 12, alpha 12
- Video modules → rank 32, alpha 16

Precedence is: `cross_modal_*` override > `audio_*` override > base `--network_dim` / `--network_alpha`.

All override flags default to `None` (no override, all modules use `--network_dim`/`--network_alpha`). Not used with LyCORIS — use the LyCORIS per-module config instead. At inference, each module's rank is read from saved weight shapes (`lora_down.shape[0]`), so no flags are needed for loading.

#### Adaptive LoRA Rank

Implemented only for standard LoRA (`networks.lora_ltx2` / `networks.lora`).
Related paper: [Not All Layers Are Created Equal: Adaptive Rank Allocation in Personalized Diffusion Models](https://arxiv.org/abs/2603.21884).

- `--network_args "adaptive_rank=True"`: Enable adaptive rank.
- `--network_args "adaptive_rank_target=<int>"`: Target effective rank. Default: each module's base rank.
- `--network_args "adaptive_rank_weight=<float>"`: Rank regularization weight. Default: `1e-4` when enabled.
- `--network_args "adaptive_rank_budget=<float>"`: Shared target for the sum of expected ranks.
- `--network_args "adaptive_rank_budget_ratio=<float>"`: Use `total_max_rank * ratio` when `adaptive_rank_budget` is unset.
- `--network_args "adaptive_rank_estimate=True"`: Use `<output_dir>/ltx2_estimate.json`. If the file is missing, it is generated from the current training args before rank allocation is applied.
- `--network_args "adaptive_rank_estimate_report=<path>"`: Override the estimate report path.
- `--network_args "adaptive_rank_hard_prune=True"`: Rebuild modules as static LoRA during training when the prune trigger fires.
- `--network_args "adaptive_rank_finalize_start=<float>"`: Convert remaining adaptive modules to static LoRA once training progress reaches this value.

Behavior:
- Without `adaptive_rank_hard_prune`, modules keep their configured base rank during training.
- Export writes standard LoRA weights. Inference reads per-module rank from weight shapes; no adaptive-rank runtime logic is required.
- `--save_state` also writes `adaptive_rank_runtime.json`. `--resume` restores adaptive/static module structure from it before loading model weights.
- Shared-budget loss uses the sum of expected ranks, not the final exported integer ranks.
- `adaptive_rank_budget` overrides `adaptive_rank_budget_ratio`.
- With `audio_dim` / `cross_modal_dim`, each module keeps its own local maximum rank.
- Estimate score lookup reads `module_scores`, or `top_modules` as fallback, keyed by `module_path`.

CLI example:
```bash
--network_dim 64 ^
--network_args "adaptive_rank=True" "adaptive_rank_target=16" "adaptive_rank_weight=1e-4"
```

Estimate-driven example:
```bash
--network_dim 64 ^
--network_args "adaptive_rank=True" "adaptive_rank_budget_ratio=0.35" "adaptive_rank_estimate=True" "adaptive_rank_hard_prune=True"
```

Notes:
- Logged metrics include `loss/adaptive_rank`, `adaptive_rank/mean_effective_rank`, `adaptive_rank/mean_expected_rank`, `adaptive_rank/mean_target_rank`, and, when a shared budget is enabled, `adaptive_rank/expected_rank_sum` and `adaptive_rank/target_budget`.

#### Per-Module LoRA Dropout

Keep the existing global LoRA dropout as the default, but optionally override it per modality through `--network_args`.

- `--network_dropout <float>`: Base dropout for all LoRA modules.
- `--network_args "audio_dropout=<float>"`: Optional dropout override for audio-only modules.
- `--network_args "video_dropout=<float>"`: Optional dropout override for non-audio video modules.
- `--network_args "cross_modal_dropout=<float>"`: Optional dropout override for cross-modal modules (`audio_to_video`, `video_to_audio`, `av_ca_*`).

Example:
```bash
--network_dropout 0.10 ^
--network_args "audio_dropout=0.15" "video_dropout=0.05" "cross_modal_dropout=0.20"
```

Result:
- Audio-only modules → dropout `0.15`
- Video-only modules → dropout `0.05`
- Cross-modal modules → dropout `0.20`
- If no per-modality override is provided, modules keep the global `--network_dropout`

Precedence is: `cross_modal_dropout` override > `audio_dropout` override for audio-only modules > `video_dropout` override for video-only modules > global `--network_dropout`.

#### Preservation & Regularization

Optional techniques that constrain how the LoRA modifies the base model. All are disabled by default with zero overhead.

**Blank Prompt Preservation** — Prevents the LoRA from altering the model's blank-prompt output (used as the CFG baseline during inference):
```bash
--blank_preservation --blank_preservation_args multiplier=1.0
```

**Differential Output Preservation (DOP)** — Prevents the LoRA from altering class-prompt output, scoping the LoRA effect to the trigger word only:
```bash
--dop --dop_args class=woman multiplier=1.0
```
The `class` parameter should be a general description without your trigger word (e.g., `woman`, `cat`, `landscape`).

**Prior Divergence** — Encourages the LoRA to produce outputs that differ from the base model on training prompts, preventing weak/timid LoRAs:
```bash
--prior_divergence --prior_divergence_args multiplier=0.1
```

**Audio DOP** — Preserves the base model's audio predictions on non-audio training steps. Requires `--ltx2_mode av`. On each non-audio batch, constructs silence audio latents, runs the transformer with LoRA OFF and ON, and minimizes MSE on the audio branch only. Zero cost on audio batches. Mutually exclusive with `--audio_silence_regularizer`.
```bash
--audio_dop --audio_dop_args multiplier=0.5
```

**TARP (Temporally Aligned RoPE and Partitioning)** — Windowed cross-attention masks that restrict each video frame to temporally nearby audio tokens (A2V) and each audio token to its nearest video frame (V2A). Enforces temporal locality in the AV cross-attention without modifying model weights. Requires `--ltx2_mode av`. From [arXiv:2603.18600](https://arxiv.org/abs/2603.18600).
```bash
--tarp --tarp_args window_multiplier=3
```
`window_multiplier` controls the A2V window size: `s = multiplier * floor(audio_tokens / video_frames)`. Default 3 (each frame sees 3x its proportional share of audio). V2A always uses nearest-neighbour (s=1).

**DCR (Dynamic Context Routing)** — Per-sample gradient detachment in cross-attention for mixed audio/video batches. When a sample lacks audio (zero-padded) or uses a clean reference (sigma=0), DCR detaches that stream's cross-attention context, preventing gradient flow through absent or reference-only streams. Forward values are unchanged; only the gradient path is masked. Requires `--ltx2_mode av`. From [arXiv:2603.18600](https://arxiv.org/abs/2603.18600).
```bash
--dcr --dcr_args reference_detach=true
```
`reference_detach` (default `true`) additionally detaches the reference stream when its timestep sigma is exactly 0.

| Technique | Extra forwards/step | Extra backwards/step | Recommended multiplier |
|-----------|-------------------|---------------------|----------------------|
| `--blank_preservation` | +2 | +1 | 0.5 - 1.0 |
| `--dop` | +2 | +1 | 0.5 - 1.0 |
| `--prior_divergence` | +1 | 0 | 0.05 - 0.1 |
| `--audio_dop` | +2 (non-audio steps only) | +1 (non-audio steps only) | 0.3 - 1.0 |
| `--tarp` | 0 | 0 | N/A (mask only) |
| `--dcr` | 0 | 0 | N/A (gradient routing) |

> [!CAUTION]
> Each preservation technique adds transformer forward passes per step. Audio DOP costs apply only on non-audio steps. TARP and DCR add no extra passes — they modify the existing forward/backward in-place.

**CREPA (Cross-frame Representation Alignment)** — Encourages temporal consistency across video frames by aligning DiT hidden states across frames via a small projector MLP. Only the projector is trained; all other modules stay frozen. CREPA uses hooks to capture intermediate features from the existing forward pass (no extra forward passes). Two modes are available: `dino` (based on [arXiv 2506.09229](https://arxiv.org/abs/2506.09229), aligns to pre-cached DINOv2 features from neighboring frames) and `backbone` (inspired by [SimpleTuner LayerSync](https://github.com/bghira/SimpleTuner), aligns to a deeper block of the same transformer).

Enable with `--crepa`. All parameters are passed via `--crepa_args` as `key=value` pairs:

```bash
accelerate launch ... ltx2_train_network.py ^
  --crepa ^
  --crepa_args mode=backbone student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2 schedule=constant warmup_steps=0 normalize=true
```

#### CLI Flags

| Flag | Type | Description |
|------|------|-------------|
| `--crepa` | store_true | Enable CREPA regularization |
| `--crepa_args` | key=value list | Configuration parameters (see table below) |

#### CREPA Parameters (`--crepa_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `backbone` | Teacher signal source: `backbone` (deeper DiT block) or `dino` (pre-cached DINOv2 features) |
| `dino_model` | `dinov2_vitb14` | DINOv2 variant for dino mode. Must match the model used during caching. Options: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` |
| `student_block_idx` | 16 | Transformer block whose hidden states are aligned (0-47 for LTX-2 48-block model) |
| `teacher_block_idx` | 32 | Deeper block providing the teacher signal (backbone mode only, must be > `student_block_idx`) |
| `lambda_crepa` | 0.1 | Loss weight for CREPA term. Recommended range: 0.05–0.5 |
| `tau` | 1.0 | Temporal neighbor decay factor. Controls how much nearby frames contribute vs distant ones |
| `num_neighbors` | 2 | Number of neighboring frames on each side (K). Frame f aligns with frames f-K..f+K |
| `schedule` | `constant` | Lambda schedule over training: `constant`, `linear` (decay to 0), or `cosine` (cosine decay to 0) |
| `warmup_steps` | 0 | Steps before CREPA loss reaches full strength (linear ramp from 0) |
| `max_steps` | 0 | Total training steps for schedule computation. Auto-filled from `--max_train_steps` if not set |
| `normalize` | `true` | L2-normalize features before computing cosine similarity |

#### Checkpoint & Resume

- The projector weights (~33M params for backbone mode) are saved as `crepa_projector.safetensors` in the output directory alongside LoRA checkpoints.
- When resuming training with `--crepa`, projector weights are automatically loaded from `<output_dir>/crepa_projector.safetensors` if the file exists.
- The projector is **not needed at inference** — it's only used during training.

#### Monitoring

CREPA adds a `loss/crepa` metric to TensorBoard/WandB logs. A healthy CREPA loss should:
- Start negative (cosine similarity is being maximized)
- Gradually decrease (more negative = stronger cross-frame alignment)
- Stabilize after warmup

#### Compatibility

- Works with block swap (`--blocks_to_swap`) — hooks fire when each block executes regardless of CPU offloading.
- Works with all preservation techniques (blank preservation, DOP, prior divergence).
- Works with gradient checkpointing.
- Projector params are included in gradient clipping alongside LoRA params.

#### Caching DINOv2 Features (Dino Mode)

Dino mode requires pre-cached DINOv2 features. Run this **after latent caching** (cache paths are derived from latent cache files). DINOv2 is not loaded during training — zero VRAM overhead.

```bash
python ltx2_cache_dino_features.py ^
  --dataset_config dataset.toml ^
  --dino_model dinov2_vitb14 ^
  --dino_batch_size 16 ^
  --device cuda ^
  --skip_existing
```

- `--dino_model`: DINOv2 variant — `dinov2_vits14` (384d), `dinov2_vitb14` (768d, default), `dinov2_vitl14` (1024d), `dinov2_vitg14` (1536d).
- `--dino_batch_size`: Frames per forward pass. Reduce if OOM (default: 16).
- `--skip_existing`: Skip items that already have cached features.

Output: `*_ltx2_dino.safetensors` files alongside your latent caches, containing per-frame patch tokens `[T, N_patches, D]`. For `dinov2_vitb14` at 518px input: `N_patches=1369`, `D=768`, so each frame adds ~2MB (float16). Disk usage scales linearly with frame count.

**Precaching preservation prompts:** Blank preservation and DOP require Gemma to encode their prompts at training startup. To avoid loading Gemma during training, precache the embeddings during the text encoder caching step:
```bash
python ltx2_cache_text_encoder_outputs.py --dataset_config ... --ltx2_checkpoint ... --gemma_root ... ^
  --precache_preservation_prompts --blank_preservation --dop --dop_class_prompt "woman"
```
Then add the `--use_precached_preservation` flag during training:
```bash
python ltx2_train_network.py ... ^
  --blank_preservation --dop --dop_args class=woman ^
  --use_precached_preservation
```
The cache file is saved to `<cache_directory>/ltx2_preservation_cache.pt` by default (same directory as your dataset cache). Use `--preservation_prompts_cache <path>` to override the location in either command. Prior divergence does not need precaching (it uses the training batch's own embeddings).

#### Self-Flow (Self-Supervised Flow Matching)

**Self-Flow** prevents the fine-tuned model from drifting away from the pretrained model's internal representations. It aligns student features (shallower block) against teacher features (deeper block) using cosine similarity, with dual-timestep noising to create a meaningful student-teacher gap. The default `teacher_mode=base` uses the **frozen pretrained model** as teacher by zeroing LoRA multipliers for the teacher forward pass — no extra VRAM overhead compared to EMA. An EMA-based teacher (`teacher_mode=ema`) is also available for LoRA-aware distillation. The optional **temporal extension** adds frame-neighbor and motion-delta losses that explicitly preserve temporal coherence. Based on [arXiv 2603.06507](https://arxiv.org/abs/2603.06507).

Enable with `--self_flow`. Supported in `--ltx2_mode video` and `--ltx2_mode av` (video branch only in AV mode). All parameters are passed via `--self_flow_args` as `key=value` pairs:

```bash
# Recommended default: base-model teacher, token-level alignment only
accelerate launch ... ltx2_train_network.py ^
  --self_flow ^
  --self_flow_args teacher_mode=base student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.1 mask_ratio=0.1 dual_timestep=true

# With temporal consistency (hybrid = frame alignment + motion delta)
accelerate launch ... ltx2_train_network.py ^
  --self_flow ^
  --self_flow_args lambda_self_flow=0.1 temporal_mode=hybrid lambda_temporal=0.1 lambda_delta=0.05 num_neighbors=2 temporal_granularity=frame
```

##### CLI Flags

| Flag | Type | Description |
|------|------|-------------|
| `--self_flow` | store_true | Enable Self-Flow regularization |
| `--self_flow_args` | key=value list | Configuration parameters (see table below) |

##### Self-Flow Parameters (`--self_flow_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher_mode` | `base` | Teacher source: `base` (frozen pretrained; zero LoRA multipliers during teacher pass, no extra VRAM), `ema` (EMA over all LoRA params), `partial_ema` (EMA over teacher block's LoRA params only) |
| `student_block_idx` | `16` | Student feature block index (0-based; overridden by `student_block_ratio` when set) |
| `teacher_block_idx` | `32` | Teacher feature block index (must be `> student_block_idx`; overridden by `teacher_block_ratio` when set) |
| `student_block_ratio` | `None` | Ratio-based student layer selection. Resolves to `floor(ratio * depth)`. Takes priority over `student_block_idx`. |
| `teacher_block_ratio` | `None` | Ratio-based teacher layer selection. Resolves to `ceil(ratio * depth)`. Takes priority over `teacher_block_idx`. |
| `student_block_stochastic_range` | `0` | Randomly vary the student capture block ±N blocks each step. `0` = fixed block. Adds regularization diversity; note a single projector is shared across all depth variants. |
| `lambda_self_flow` | `0.1` | Loss weight for the video token-level representation alignment term |
| `lambda_audio` | `0.0` | Loss weight for audio representation alignment. When `> 0`, captures audio hidden states from the same student/teacher blocks and aligns them via a separate audio projector MLP. Requires `--ltx2_mode av`. `0` = disabled (backward compatible). |
| `mask_ratio` | `0.10` | Token mask ratio for dual-timestep mixing. Valid range: `[0.0, 0.5]` |
| `frame_level_mask` | `false` | When `true`, mask whole latent frames instead of individual tokens. More semantically coherent masking for video. |
| `mask_focus_loss` | `false` | When `true`, compute the representation loss only on masked (higher-noise) tokens. Default: loss over all tokens. |
| `max_loss` | `0.0` | Cap Self-Flow loss magnitude by rescaling if total loss exceeds this value. `0` = disabled. Useful to prevent Self-Flow from dominating the main task loss early in training. |
| `temporal_mode` | `off` | Temporal extension mode: `off`, `frame`, `delta`, or `hybrid` |
| `lambda_temporal` | `0.0` | Loss weight for frame-level temporal neighbor alignment |
| `lambda_delta` | `0.0` | Loss weight for frame-delta alignment (motion consistency) |
| `temporal_tau` | `1.0` | Neighbor decay factor for `frame` / `hybrid` temporal alignment |
| `num_neighbors` | `2` | Number of temporal neighbors on each side used by `frame` / `hybrid` mode |
| `temporal_granularity` | `frame` | Temporal loss granularity: `frame` (mean-pooled per frame) or `patch` (preserve spatial tokens) |
| `patch_spatial_radius` | `0` | In `temporal_granularity=patch`, local spatial neighborhood radius for teacher patch matching (`0` = strict same-patch only) |
| `patch_match_mode` | `hard` | Patch-neighborhood matching mode: `hard` (best patch in window) or `soft` (softmax-weighted neighborhood match) |
| `patch_match_temperature` | `0.1` | Soft neighborhood matching temperature when `patch_match_mode=soft` |
| `delta_num_steps` | `1` | Number of temporal delta steps included in the delta loss (`1` = adjacent frames only) |
| `motion_weighting` | `none` | Temporal weighting mode: `none` or `teacher_delta` |
| `motion_weight_strength` | `0.0` | Strength of teacher-delta motion weighting for temporal terms |
| `temporal_schedule` | `constant` | Schedule applied to **all** Self-Flow lambdas (`lambda_self_flow`, `lambda_temporal`, `lambda_delta`): `constant`, `linear` decay, or `cosine` decay |
| `temporal_warmup_steps` | `0` | Steps to linearly ramp all lambdas up from zero to full weight |
| `temporal_max_steps` | `0` | Steps at which `linear` / `cosine` decay reaches zero. `0` = no decay |
| `teacher_momentum` | `0.999` | EMA momentum for teacher updates (`ema` / `partial_ema` modes only). Valid range: `[0.0, 1.0)` |
| `teacher_update_interval` | `1` | Update EMA teacher every N optimizer steps |
| `projector_hidden_multiplier` | `1` | Projector hidden width multiplier vs model inner dim |
| `projector_activation` | `silu` | Projector MLP activation function: `silu` or `gelu` |
| `projector_lr` | `None` | Optional projector-specific learning rate. Defaults to `--learning_rate` when unset |
| `loss_type` | `negative_cosine` | `negative_cosine` or `one_minus_cosine` |
| `dual_timestep` | `true` | Enable dual-timestep noising |
| `tokenwise_timestep` | `true` | Use per-token timesteps (otherwise per-sample averaged timestep) |
| `offload_teacher_features` | `false` | Offload cached teacher features to CPU to reduce VRAM |
| `offload_teacher_params` | `false` | Offload EMA teacher parameters to CPU (saves VRAM, slower teacher forward pass; `ema` / `partial_ema` only) |

##### Notes

- Supported modes: `--ltx2_mode video`, `--ltx2_mode av`. In AV mode, video alignment is always active when `lambda_self_flow > 0`; audio alignment is active when `lambda_audio > 0`.
- Image-like training is supported through single-frame samples in `--ltx2_mode video` (set `temporal_mode=off` unless you intentionally want temporal terms to be inactive on image batches).
- Cost: one extra teacher forward pass per train step. `teacher_mode=base` requires no extra VRAM since it reuses the existing model with LoRA multipliers zeroed.
- Teacher modes: `base` gives the largest student-teacher gap (pretrained vs LoRA-finetuned); `ema` / `partial_ema` give a moving target that shrinks as training converges.
- Temporal extension: when `temporal_mode != off`, Self-Flow reshapes hidden states into latent frames and adds frame-neighbor and/or frame-delta consistency losses on top of the base token alignment loss.
- Granularity: `temporal_granularity=frame` uses mean-pooled per-frame features (cheaper, coarser). `temporal_granularity=patch` keeps spatial tokens for stronger temporal matching.
- Local patch matching: when `temporal_granularity=patch` and `patch_spatial_radius > 0`, each student patch can align to the best teacher patch inside a local spatial window, which is more tolerant to small motion and camera drift than strict same-patch matching.
- Soft matching: `patch_match_mode=soft` replaces hard local best-match selection with softmax-weighted neighborhood matching for smoother gradients.
- Multi-step motion: `delta_num_steps > 1` extends the delta loss beyond adjacent frames using exponentially decayed step weights.
- Motion-aware weighting: `motion_weighting=teacher_delta` upweights temporally active teacher regions, focusing the temporal loss on moving content.
- Scheduling: `temporal_schedule`, `temporal_warmup_steps`, and `temporal_max_steps` apply to **all three** lambdas — `lambda_self_flow`, `lambda_temporal`, and `lambda_delta` — uniformly.
- State files (Accelerate `*-state` folder): `self_flow_projector.safetensors`, `self_flow_teacher_ema.safetensors` (EMA state only saved when `teacher_mode=ema` or `partial_ema`).
- Resume: both state files are loaded automatically when present. Loading EMA state with `teacher_mode=base` emits a warning and is ignored.
- Logged metrics: `loss/self_flow`, `self_flow/cosine`, `self_flow/frame_cosine`, `self_flow/delta_cosine`, `self_flow/lambda_self_flow`, `self_flow/lambda_temporal`, `self_flow/lambda_delta`, `self_flow/masked_token_ratio`, `self_flow/tau_mean`, `self_flow/tau_min_mean`.

#### HFATO (High-Frequency Awareness Training Objective)

Adapted from [ViBe (arXiv 2603.23326)](https://arxiv.org/abs/2603.23326). Experimental — not yet validated on LTX-2.

**HFATO** is a training objective designed for image-only fine-tuning of video models. Before adding noise, clean latents are spatially degraded via downsample-upsample, destroying high-frequency details. The model is then supervised to reconstruct the original clean latents (x₀-prediction loss instead of standard velocity loss). Can be combined with the Relay LoRA workflow below for two-stage image-only training.

Enable with `--hfato`. Parameters are passed via `--hfato_args` as `key=value` pairs. Incompatible with `--ic_lora_strategy v2v`.

```bash
accelerate launch ... ltx2_train_network.py ^
  --hfato ^
  --hfato_args scale_factor=0.5
```

##### CLI Flags

| Flag | Type | Description |
|------|------|-------------|
| `--hfato` | store_true | Enable HFATO loss |
| `--hfato_args` | key=value list | Configuration parameters (see table below) |

##### HFATO Parameters (`--hfato_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_factor` | `0.5` | Spatial downsample ratio. `0.5` = halve each spatial dimension. Lower values destroy more high-frequency info and force stronger reconstruction. `0.25` is more aggressive. |
| `interpolation` | `bilinear` | Interpolation mode for downsample-upsample: `bilinear`, `nearest`, or `bicubic` |
| `probability` | `1.0` | Per-step probability of applying HFATO. `1.0` = always. Values `< 1.0` mix HFATO and standard flow matching steps. |

##### Relay LoRA Workflow (Image-Only Training)

Two-stage training strategy for adapting a video model using only images:

1. **Stage 1 (Modality Bridge)**: Train a LoRA on low-resolution images using standard flow matching. Bridges the image-video modality gap.
2. **Merge**: Merge the Stage 1 LoRA into the base model checkpoint.
3. **Stage 2 (Detail Enhancement)**: Train a new LoRA on high-resolution images with `--hfato`, using the merged checkpoint.
4. **Inference**: Load the original base model with only the Stage 2 LoRA. Stage 1 is discarded.

```bash
# Stage 1: standard LoRA on low-res images
accelerate launch ... ltx2_train_network.py ^
  --ltx2_checkpoint base_model.safetensors ^
  --network_dim 32 --output_dir stage1_output/

# Merge Stage 1 into base
python ltx2_merge_lora_to_model.py ^
  --dit base_model.safetensors ^
  --lora_weight stage1_output/last.safetensors ^
  --save_merged_model merged_model.safetensors

# Stage 2: HFATO on high-res images using merged base
accelerate launch ... ltx2_train_network.py ^
  --ltx2_checkpoint merged_model.safetensors ^
  --hfato --hfato_args scale_factor=0.5 ^
  --network_dim 32 --output_dir stage2_output/

# Inference: original base + Stage 2 LoRA only
python ltx2_generate_video.py ^
  --ltx2_checkpoint base_model.safetensors ^
  --lora_weight stage2_output/last.safetensors
```

The resulting LoRA is standard — no inference pipeline changes. Discarding the Stage 1 bridge at inference acts as an implicit regularizer that limits how far the model drifts from its pretrained video priors.

HFATO can also be used standalone (without relay) as a spatial detail objective for image or video training.

#### Standalone Inference Overrides

`ltx2_generate_video.py` accepts a few standalone-inference-only overrides that are not part of the training sample table:

- `--vae`: Use a separate VAE checkpoint for inference. If omitted, `--ltx2_checkpoint` is used for both DiT and VAE loading.
- `--vae_dtype`: Override the VAE runtime dtype for inference. If omitted, the script uses its default VAE dtype (`bfloat16`).
- `--reference_image`: Apply one global I2V reference image to all prompts in the current inference run.
- `--reference_video`: Apply one global V2V reference video to all prompts in the current inference run.
- If both `--reference_image` and `--reference_video` are supplied, `--reference_video` takes priority.
- Global `--reference_image` / `--reference_video` overrides replace conflicting per-prompt `image_path` / `v2v_ref_path` entries loaded from prompt files, and also clear any cached reference latents tied to those prompt entries before sampling.
- If the path passed to `--reference_image` has a video filename extension, the script treats it as a V2V reference and routes it through the video-reference path.

#### Audio Quality Metrics

Enable with `--audio_metrics`. All logic in `audio_metrics.py`. Disabled by default with zero overhead.

**Per-step** (latent-space, enabled by default with `--audio_metrics`):

| Key | Description |
|-----|-------------|
| `audio_metrics/latent_fd` | Running Frechet distance between pred/target latent distributions (every 50 steps) |
| `audio_metrics/temporal_coherence` | Cosine similarity between adjacent audio latent frames |
| `audio_metrics/av_latent_sync` | Pearson correlation between audio and video per-frame energy |

**Periodic** (mel-space, opt-in — decodes 1 sample per batch through AudioDecoder):
```bash
--audio_metrics --audio_metrics_args mel_metrics=true mel_compute_every=100
```

| Key | Description |
|-----|-------------|
| `audio_metrics/spectral_convergence` | `\|\|S_pred - S_target\|\|_F / \|\|S_target\|\|_F` |
| `audio_metrics/mcd_db` | Mel Cepstral Distortion (13 DCT coefficients, dB) |
| `audio_metrics/log_spectral_distance_db` | Per-frame log-spectral distance (dB) |

**Sampling-time** (embedding-space, opt-in — runs on generated waveforms during `sample_images`):
```bash
--audio_metrics --audio_metrics_args clap_similarity=true av_onset_alignment=true
```

| Key | Description | Requires |
|-----|-------------|----------|
| `sample_audio/clap_similarity` | CLAP audio-text cosine similarity | transformers (already a dep) |
| `sample_audio/av_onset_alignment` | Correlation between audio energy onsets and video motion | None |

CLAP model is lazy-loaded on first sample, offloaded to CPU between uses.

#### Timestep Sampling

See also the [timestep bucketing documentation](./advanced_config.md) for advanced timestep bucketing options.

- `--timestep_sampling shifted_logit_normal`: Default LTX-2 method. Uses a shifted logit-normal distribution where the shift is computed from latent sequence length. In normal video/AV training this means `latent_frames × latent_height × latent_width`; only `--ltx2_mode audio` uses the audio-only sequence-length path described below.
- `--timestep_sampling uniform`: Uniform sampling from [0, 1].
- `--logit_std`: Standard deviation for the logit-normal distribution (default: 1.0). Only used with `shifted_logit_normal`.
- `--min_timestep` / `--max_timestep`: Optional timestep range constraints.
- `--shifted_logit_mode legacy|stretched`: Sigma sampler variant (default: auto by `--ltx_version`; 2.0→`legacy`, 2.3→`stretched`).
  - `legacy`: `sigmoid(N(shift, std))`. Original behavior.
  - `stretched`: Normalizes samples between the 0.5th and 99.9th percentiles of the distribution, reflects values below `eps` for numerical stability, and replaces a fraction of samples with uniform draws to prevent distribution collapse at high token counts.
- `--shifted_logit_eps`: Reflection floor and uniform lower bound for `stretched` mode (default: `1e-3`).
- `--shifted_logit_uniform_prob`: Fraction of samples replaced with uniform `[eps, 1]` draws (default: `0.1`).
- `--shifted_logit_shift`: Override the auto-calculated shift value. Lower values (e.g., `0.0`) produce a symmetric distribution centered on medium noise (σ≈0.5) for learning fine details. Higher values (e.g., `2.0`) heavily right-skew the distribution toward high noise (σ≈0.9+) for learning global structure. If unset, it is computed dynamically from sequence length. In the current trainer implementation, non-audio training uses the raw linear formula below (so short or long sequences can fall outside the anchor values), while `--ltx2_mode audio` clamps the auto-computed shift to `[0.95, 2.05]`.

> [!NOTE]
> The `shifted_logit_normal` auto-shift uses a linear formula anchored at 0.95 for 1024 tokens and 2.05 for 4096 tokens, based on sequence length. The current non-audio trainer extrapolates this formula outside those anchor points for shorter/longer sequences; for example, a single 768x768 image has latent sequence length `1 x (768/32) x (768/32) = 576`, which gives shift `0.7896`. In `--ltx2_mode audio`, the auto-computed shift is clamped to `[0.95, 2.05]`.
> In `--ltx2_mode audio`, `shifted_logit_normal` still needs a sequence length to compute the shift, but there is no real video spatial dimension. Using full video resolution would inflate the sequence length and skew the shift upward. Instead, `--audio_only_sequence_resolution` (default `64`) provides a small fixed spatial footprint (4 tokens/frame), keeping the shift dominated by the temporal dimension (audio duration/FPS) which actually matters.
> In joint AV training (`--ltx2_mode av`), the auto shift still comes from the video latent geometry; the presence of audio latents does not change the shift calculation.

#### LoRA Targets

Use `--lora_target_preset` to control which layers LoRA targets. For custom layer patterns and `--network_args` format, see the [LoRA documentation](./advanced_config.md#lora):

| Preset | Layers | Modality scope | Use Case |
|--------|--------|----------------|----------|
| `t2v` (default) | All attention (`to_q`, `to_k`, `to_v`, `to_out.0`) | Video + audio + cross-modal | Text-to-video, matches official LTX-2 trainer |
| `v2v` | All attention + video FFN + audio FFN | Video + audio + cross-modal | Video-to-video / IC-LoRA style |
| `video_sa` | Video self-attention (`attn1`) | Video only | Spatially-aligned controls (depth, pose, canny, inpaint) |
| `video_sa_ff` | Video self-attention + video FFN (`attn1`, `ff`) | Video only | Controls needing more capacity (local edit, cut-on-action) |
| `video_sa_ca_ff` | Video self-attention + cross-attention + video FFN (`attn1`, `attn2`, `ff`) | Video only | Text-guided controls (video detailing, camera-from-image, sparse tracks) |
| `audio` | Audio attn/FFN + `video_to_audio_attn` | Audio only | Audio-only training (auto-selected when `--ltx2_mode audio`) |
| `audio_ref_only_ic` | Audio attn/FFN + bidirectional AV cross-modal | Audio + cross-modal | Audio-reference IC-LoRA |
| `av_ic` | All attention + video FFN + audio FFN (same as `v2v`) | Video + audio + cross-modal | Joint AV IC-LoRA. Use `--av_cross_attention_mode` for directional variants and `--av_multi_ref` when configuring a multi-reference AV IC run |
| `video_ref_only_av` | All attention + video FFN + audio FFN (same as `v2v`) | Video + audio + cross-modal | AV training with reference video only; target audio is still generated |
| `full` | All linear layers for LoRA targeting | Video + audio + cross-modal | Maximum expressiveness, larger file size |

**Modality scope matters when training on an AV checkpoint.** The `t2v`, `v2v`, `av_ic`, and `full` presets create LoRA weights for audio and cross-modal layers. If those layers receive no audio training signal (e.g., image/video-only dataset), the LoRA weights for audio modules are initialized but never meaningfully updated — applying such a LoRA can degrade the base model's audio capabilities. Use a `video_*` preset to restrict LoRA to video-branch modules only, leaving audio layers completely untouched. Connector layers (`Embeddings1DConnector`) are excluded by default; use `--train_connectors` to include them (see below).

To use custom layer patterns instead of a preset, use `--network_args`:
```bash
--network_args "include_patterns=['.*\.to_k$','.*\.to_q$','.*\.to_v$','.*\.to_out\.0$','.*\.ff\.net\.0\.proj$','.*\.ff\.net\.2$']"
```
Custom `include_patterns` override any preset.
When `include_patterns` is set (either explicitly or via a preset), only modules matching at least one pattern are targeted (strict whitelist behavior). Use `--lora_target_preset full` to target all linear layers.

#### LoRA Target Estimation (`ltx2_estimate.py`)

`ltx2_estimate.py` runs the LTX forward/loss path on cached training batches and accumulates squared gradients ("Fisher-style importance") for LoRA-targetable weights.

- It uses `setup_parser_common()` plus `ltx2_setup_parser()`, so the normal LTX argument surface is available. In the estimator path, attention backend selection (`--sdpa`, `--flash_attn`, `--flash3`, `--xformers`), `--blocks_to_swap`, `--gradient_checkpointing`, `--blockwise_checkpointing`, `--compile`, `--fp8_base` / `--fp8_scaled`, `--nf4_base`, and `--split_attn` are applied.
- It requires `--dataset_config` and cached dataset items. If the dataset group has no training items, it exits with `No training items found in the dataset. Create latent/text caches first.`
- It keeps up to `--estimation_batches` batches. Batches without 5D `latents` are skipped.
- If `--network_weights` is set, the estimator attaches that LoRA to the transformer and scores the LoRA weights from the attached network (`candidate_source = "network"`).
- If `--network_weights` is not set, it scores LoRA-targetable linear weights from the transformer itself (`candidate_source = "base_model"`).
- If `--base_weights` is set, those weights are merged into the transformer before estimation.
- Unless `--estimation_keep_caption_dropout` is set, the estimator temporarily forces `caption_dropout_rate = 0`.
- `--estimation_block_window` only affects the base-model path: it groups candidate weights by transformer block and enables one window of blocks per backward pass. When `--network_weights` is used, all network candidates are scored in one pass.

Example:

```bash
python ltx2_estimate.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
  --mixed_precision bf16 ^
  --ltx_version 2.3 ^
  --ltx2_mode av ^
  --network_module networks.lora_ltx2 ^
  --network_weights output/your_lora.safetensors ^
  --estimation_batches 8 ^
  --estimation_output output/ltx2_estimate.json ^
  --flash_attn --fp8_base --fp8_scaled --blocks_to_swap 10 --gradient_checkpointing
```

The output is a JSON report written to `--estimation_output` or, if omitted, `<output_dir>/ltx2_estimate.json`.

- `meta`: run configuration, timing, applied / merged weights, and `candidate_source`
- `summary`: `candidate_modules`, `candidate_params`, `total_fisher_sum`, and `recommended_preset`
- `family_scores`: aggregate scores by module family such as `video_self_attn` and `video_cross_attn`
- `preset_scores`: aggregate scores for the preset candidates available in the current `ltx_mode`
- `module_scores`: per-module score rows keyed by `module_path`
- `top_modules`: highest-ranked individual weights

`recommended_preset` is selected as follows:

- Pick the smallest preset whose `fisher_share` reaches `--estimation_target_coverage`
- If no preset reaches that threshold, pick the preset with the highest `efficiency`

#### Connector LoRA (`--train_connectors`)

Text connectors are 8-layer transformer blocks (in LTX-2.3) between Gemma and the denoising transformer. They transform text embeddings before they reach the denoising model. `--train_connectors` includes these modules in LoRA training alongside the main transformer.

**Usage:** Cache with `--cache_before_connector`, train with `--train_connectors`. The same `--lora_target_preset` patterns apply to both transformer and connector layers. Connector and transformer LoRA weights are saved in one file. At inference and in ComfyUI (after `convert_lora_to_comfy.py`), connector weights are auto-detected and applied.

**Notes:** Adds ~3.8 GB VRAM (bf16) for the frozen connector weights. Not compatible with LyCORIS. Connectors have `attn1` and `ff` only (no `attn2`).

#### IC-LoRA / Video-to-Video Training

IC-LoRA (In-Context LoRA) trains the model to generate video conditioned on a reference image or video.

Reference frames are encoded as clean latent tokens (timestep=0) and concatenated with noisy target tokens during training. The model attends across both sequences, using the reference as conditioning context. At inference, the same concatenation scheme is applied. Position embeddings are computed separately for reference and target, allowing different spatial resolutions via `--reference_downscale`.

##### Step 1: Prepare Dataset

Create a dataset with a matching reference/source directory. For both video and image IC-LoRA datasets, use `reference_directory`. Each reference file must share the same filename stem as its corresponding training sample:

```
videos/                    references/
  scene_001.mp4              scene_001.png     # reference for scene_001
  scene_002.mp4              scene_002.jpg
  scene_003.mp4              scene_003.mp4     # video references also work
```

References can be images (single frame) or videos (multiple frames).

##### Step 2: Dataset Config

Add `reference_cache_directory` plus the matching source directory key to your TOML config:

```toml
[general]
resolution = [768, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
cache_directory = "cache"
reference_cache_directory = "cache_ref"

[[datasets]]
video_directory = "videos"
reference_directory = "references"
target_frames = [1, 17, 33]
```

For image datasets, use `reference_directory` as well:

```toml
[[datasets]]
image_directory = "targets"
reference_directory = "references"
reference_cache_directory = "cache_ref"
```

For multi-reference `av_ic`, use the plural dataset keys instead:

```toml
[[datasets]]
video_directory = "videos"
reference_directories = ["references_a", "references_b"]
reference_cache_directories = ["cache_ref_a", "cache_ref_b"]
reference_audio_directories = ["reference_audio_a", "reference_audio_b"]
reference_audio_cache_directories = ["cache_ref_audio_a", "cache_ref_audio_b"]
target_frames = [33]
```

In the dashboard, these map to the Advanced dataset fields:
- `Reference Cache Dir` + `Extra Ref Cache Dirs`
- `Reference Directory` + `Extra Reference Dirs`
- `Ref Audio Cache Dir` + `Extra Ref Audio Cache Dirs`
- `Ref Audio Directory` + `Extra Ref Audio Dirs`

In the training dashboard, AV IC behavior is configured from the Advanced LoRA section:
- `IC-LoRA Strategy = av_ic`
- `AV Cross-Attn` for `both` / `a2v_only` / `v2a_only` / `none`
- `Multi-Ref AV` when the dataset uses the plural reference directory fields above

##### Step 3: Cache Latents

Cache both video latents and reference latents in one step:

```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --device cuda ^
  --vae_dtype bf16
```

Reference latents are automatically cached to `reference_cache_directory` when `reference_directory` is configured.

**Downscaled references** (`--reference_downscale`):
```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --reference_downscale 2 ^
  --device cuda
```

`--reference_downscale 2` encodes references at half spatial resolution (e.g., 384px for 768px target). Position embeddings on the reference spatial axes are scaled by the factor so they map into the target coordinate space.

##### Step 4: Cache Text Encoder Outputs

Same as standard training — no special flags needed:
```bash
python ltx2_cache_text_encoder_outputs.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --gemma_root /path/to/gemma ^
  --gemma_load_in_8bit ^
  --device cuda
```

##### Step 5: Train

Use `--lora_target_preset v2v` (targets attention + FFN layers):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --fp8_base --fp8_scaled ^
  --blocks_to_swap 10 ^
  --sdpa ^
  --gradient_checkpointing ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 --network_alpha 32 ^
  --lora_target_preset v2v ^
  --ltx2_first_frame_conditioning_p 0.2 ^
  --timestep_sampling shifted_logit_normal ^
  --learning_rate 1e-4 ^
  --sample_at_first ^
  --sample_every_n_epochs 5 ^
  --sample_prompts sampling_prompts.txt ^
  --sample_include_reference ^
  --output_dir output ^
  --output_name ltx2_ic_lora
```

If you used `--reference_downscale` during caching, also pass it during training:
```bash
  --reference_downscale 2
```

##### Step 6: Sample Prompts

Use `--v <path>` in your sampling prompts file to specify the V2V reference for each prompt. Both images and videos are supported:

```
--v references/scene_001.png A woman walking through a forest --n blurry, low quality
--v references/scene_002.mp4 A cat sitting on a windowsill --n distorted
```

The `--sample_include_reference` flag shows the reference side-by-side with the generated output in validation videos.

##### IC-LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--reference_downscale` | 1 | Spatial downscale factor for references (1=same res, 2=half) |
| `--reference_frames` | 1 | Number of reference frames for V2V (images are repeated to fill this count) |
| `--ltx2_first_frame_conditioning_p` | 0.1 | Probability of also conditioning on the first target frame during training. No effect for single-frame (image) samples |
| `--sample_include_reference` | off | Show reference side-by-side with generated output in sample videos |
| `--lora_target_preset v2v` | — | Targets attention + FFN layers (recommended for IC-LoRA) |

##### Dataset Config Options

| Option | Type | Description |
|--------|------|-------------|
| `reference_directory` | string | Path to reference images/videos for IC-LoRA datasets (matched by filename stem) |
| `reference_directories` | array[string] | Optional multi-reference variant of `reference_directory`; use one entry per reference stream |
| `reference_cache_directory` | string | Output directory for cached reference latents |
| `reference_cache_directories` | array[string] | Optional multi-reference variant of `reference_cache_directory`; count must match `reference_directories` |

##### Notes

- **First-frame conditioning** (`--ltx2_first_frame_conditioning_p`): Randomly conditions on the first target frame in addition to the reference. Only applied during training; inference always denoises the full target. Has no effect for single-frame (image-only) samples — the code skips conditioning when `num_frames == 1` since there are no subsequent frames to generate.
- **Multi-frame references**: Supported but increase VRAM usage proportionally to the number of reference tokens.
- **Multi-reference datasets**: `av_ic` can consume multiple references directly from dataset TOML via `reference_directories` + `reference_cache_directories` (and the audio equivalents below). The list lengths must match. `--av_multi_ref` exposes that intent in training metadata/UI.
- **Multi-subject references**: The VAE compresses 8 frames into 1 temporal latent via `SpaceToDepthDownsample`, which pairs consecutive frames and averages their features. Subjects sharing the same 8-frame group are blended and lose individual identity. To keep N subjects separated, structure your reference video as: frame 1 = Subject A, frames 2–9 = Subject B (repeated 8×), frames 10–17 = Subject C (repeated 8×), etc. Total frames: `1 + 8×(N−1)`. Set `--reference_frames` to match. Frame 1 gets its own latent due to causal padding in the encoder; each subsequent 8-frame block produces one additional latent.
- **Video-only**: IC-LoRA requires `--ltx2_mode video`. Audio-video mode is not supported for v2v training.
- **Downscale factor metadata**: Saved in LoRA safetensors as `ss_reference_downscale_factor` when factor != 1.
- **Two-stage inference**: Not supported with V2V; a warning is emitted and the reference is ignored.

#### Audio-Reference IC-LoRA

> This approach is based on [ID-LoRA](https://github.com/ID-LoRA/ID-LoRA), adapted for audio-video conditioning in the LTX-2 transformer.

Trains a LoRA using in-context audio-reference conditioning. Reference audio latents (clean, timestep=0) are concatenated with noisy target audio latents during training. Loss is computed only on the target portion. In AV mode the LoRA targets audio self/cross-attention, audio FFN, and bidirectional audio-video cross-modal attention layers; in audio-only mode the `audio` preset is auto-selected, which omits cross-modal layers that connect to the (dummy) video branch.

Supported modes:
- **`--ltx2_mode av`** — full audio-video model; trains both video and audio IC-LoRA layers.
- **`--ltx2_mode audio`** — audio-only mode; trains only audio layers (video is a dummy zero tensor). `--lora_target_preset audio` is auto-selected (cross-modal layers that affect the dummy video branch are omitted).

##### Recommended settings

The ID-LoRA reference configuration enables all of the following. A warning is emitted at training start if any are missing:

| Setting | Recommended value | Why |
|---------|-------------------|-----|
| `--ltx2_first_frame_conditioning_p` | `0.9` | Face identity comes from the first frame; voice identity comes from the reference LoRA. Without this, face identity is uncontrolled. |
| `--audio_ref_use_negative_positions` | enabled | Clean positional separation between reference and target in RoPE space. |
| `--audio_ref_mask_cross_attention_to_reference` | enabled | Forces video to sync with target audio only (not reference). AV mode only. |
| `--audio_ref_mask_reference_from_text_attention` | enabled | Prevents reference audio from attending to text describing the target speech. |
| `--timestep_sampling` | `shifted_logit_normal` | Timestep distribution used by ID-LoRA. |
| `--network_dim` / `--network_alpha` | `128` / `128` | LoRA rank used by ID-LoRA. |

> **Inference note**: Attention masks (`--audio_ref_mask_cross_attention_to_reference` and `--audio_ref_mask_reference_from_text_attention`) are **training scaffolding only**. They are automatically disabled during sampling/inference, matching the ID-LoRA reference which explicitly turns masks off at validation time. The masks force the model to learn proper attention patterns during training; at inference time the LoRA weights have already internalized the separation, so masks are unnecessary.

##### How it works

1. Reference audio is encoded to latents and concatenated with noisy target audio along the temporal axis.
2. Reference tokens receive timestep=0 (no noise); target tokens receive the sampled sigma.
3. Loss is masked to exclude the reference portion — the model only learns to predict the target.
4. Three optional attention overrides control how the reference interacts with the rest of the model:
   - **Negative positions**: shifts reference tokens into negative RoPE time, creating clean positional separation from target tokens.
   - **A2V cross-attention mask**: blocks video from attending to reference audio (video syncs with target audio only).
   - **Text attention mask**: blocks reference audio from attending to text (reference provides identity, not content).

##### Step 1: Prepare Data

Organize training videos with matching reference audio files (same filename stem):

```
videos/                    reference_audio/
  speaker_001.mp4            speaker_001.wav    # reference clip for speaker_001
  speaker_002.mp4            speaker_002.flac
```

Reference audio files are matched to training videos by filename stem.

##### Step 2: Dataset Config

```toml
[general]
resolution = [768, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
cache_directory = "cache"
reference_audio_cache_directory = "cache_ref_audio"
separate_audio_buckets = true

[[datasets]]
video_directory = "videos"
reference_audio_directory = "reference_audio"
target_frames = [1, 17, 33]
```

##### Step 3: Cache Latents

```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltxav-2.safetensors ^
  --ltx2_mode av ^
  --device cuda ^
  --vae_dtype bf16
```

For audio-only mode, replace `--ltx2_mode av` with `--ltx2_mode audio` (no video latents are cached, only audio and reference audio latents).

Reference audio latents are automatically cached to `reference_audio_cache_directory`.

##### Step 4: Cache Text Encoder Outputs

No special flags — use whichever mode you are training:

```bash
python ltx2_cache_text_encoder_outputs.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltxav-2.safetensors ^
  --ltx2_mode av ^
  --gemma_root /path/to/gemma ^
  --gemma_load_in_8bit ^
  --device cuda
```

For audio-only mode, replace `--ltx2_mode av` with `--ltx2_mode audio`.

##### Step 5: Train

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltxav-2.safetensors ^
  --ltx2_mode av ^
  --fp8_base --fp8_scaled ^
  --blocks_to_swap 10 ^
  --sdpa ^
  --gradient_checkpointing ^
  --network_module networks.lora_ltx2 ^
  --network_dim 128 --network_alpha 128 ^
  --lora_target_preset audio_ref_only_ic ^
  --audio_ref_use_negative_positions ^
  --audio_ref_mask_cross_attention_to_reference ^
  --audio_ref_mask_reference_from_text_attention ^
  --ltx2_first_frame_conditioning_p 0.9 ^
  --timestep_sampling shifted_logit_normal ^
  --learning_rate 2e-4 ^
  --sample_at_first ^
  --sample_every_n_epochs 5 ^
  --sample_prompts sampling_prompts.txt ^
  --output_dir output ^
  --output_name ltx2_audio_ref_ic_lora
```

**Audio-only mode** — replace `--ltx2_mode av` with `--ltx2_mode audio` and omit `--lora_target_preset` (auto-selected as `audio`):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltxav-2.safetensors ^
  --ltx2_mode audio ^
  --ic_lora_strategy audio_ref_only_ic ^
  --fp8_base --fp8_scaled ^
  --blocks_to_swap 10 ^
  --sdpa ^
  --gradient_checkpointing ^
  --network_module networks.lora_ltx2 ^
  --network_dim 128 --network_alpha 128 ^
  --audio_ref_use_negative_positions ^
  --audio_ref_mask_reference_from_text_attention ^
  --timestep_sampling shifted_logit_normal ^
  --learning_rate 2e-4 ^
  --sample_at_first ^
  --sample_every_n_epochs 5 ^
  --sample_prompts sampling_prompts.txt ^
  --output_dir output ^
  --output_name ltx2_audio_ref_ic_lora_audioonly
```

##### Step 6: Sample Prompts

Use `--ra <path>` in your sampling prompts file to specify the reference audio:

```
--ra reference_audio/speaker_001.wav A person speaking about nature --n blurry, low quality
--ra reference_audio/speaker_002.flac A woman laughing in a park
```

Reference audio latents are precached automatically when using `--precache_sample_latents` during latent caching.

##### Audio-Reference IC-LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ic_lora_strategy audio_ref_only_ic` | auto | Activates audio-reference IC-LoRA mode (auto-inferred from `--lora_target_preset audio_ref_only_ic`) |
| `--lora_target_preset audio_ref_only_ic` | — | Targets audio attn/FFN + bidirectional AV cross-modal layers |
| `--audio_ref_use_negative_positions` | off | Place reference audio in negative RoPE time for positional separation |
| `--audio_ref_mask_cross_attention_to_reference` | off | Block video from attending to reference audio tokens (AV mode only; no effect in audio-only mode) |
| `--audio_ref_mask_reference_from_text_attention` | off | Block reference audio from attending to text tokens (`av_ic`: currently unsupported and ignored) |
| `--audio_ref_identity_guidance_scale` | 0.0 | Override CFG scale for target-audio branch during `audio_ref_only_ic` sampling (0 = use standard guidance scale) |

##### Dataset Config Options

| Option | Type | Description |
|--------|------|-------------|
| `reference_audio_directory` | string | Path to reference audio files (matched by filename stem) |
| `reference_audio_directories` | array[string] | Optional multi-reference variant of `reference_audio_directory`; use one entry per reference stream |
| `reference_audio_cache_directory` | string | Output directory for cached reference audio latents |
| `reference_audio_cache_directories` | array[string] | Optional multi-reference variant of `reference_audio_cache_directory`; count must match `reference_audio_directories` |

##### Notes

- **Checkpoint**: requires an LTXAV checkpoint for both `--ltx2_mode av` and `--ltx2_mode audio`.
- **Bucket separation**: `separate_audio_buckets = true` keeps audio/non-audio items in separate batches (avoids shape mismatches in collation).
- **Attention masks are training-only**: `--audio_ref_mask_cross_attention_to_reference` and `--audio_ref_mask_reference_from_text_attention` are applied only during training. They are automatically disabled during sampling/inference to match the ID-LoRA reference (which explicitly sets both to `false` during validation). Negative position overrides are always applied.
- **`av_ic` limitation**: `--audio_ref_mask_reference_from_text_attention` is not currently supported in `av_ic` because the Modality API uses a 2D `context_mask`; the trainer warns and ignores this flag.
- **AV cross-attention modes**: `--av_cross_attention_mode both` is the default `av_ic` behavior. Use `a2v_only` for audio-to-video only, `v2a_only` for video-to-audio only, or `none` to disable AV cross-modal attention while keeping the rest of `av_ic` intact. All require `--ltx2_mode av`.
- **Multi-reference `av_ic`**: accepts multiple reference latents when they are provided as stacked tensors or extra `ref_*` entries, and concatenates them before conditioning. This keeps the implementation compatible with the existing single-reference path while allowing richer identity/style aggregation. `--av_multi_ref` is the explicit training-side toggle for this setup.
- **`video_ref_only_av`**: requires `--ltx2_mode av`, uses reference video only, and keeps the audio branch target-only. This is useful when you want identity/motion conditioning from video without requiring reference audio for every sample.
- **First-frame conditioning is critical**: without `--ltx2_first_frame_conditioning_p 0.9`, the model cannot learn face identity from the first frame. A warning is emitted if this is not set in AV mode.
- `--ic_lora_strategy auto` (default) infers the strategy from `--lora_target_preset` via `infer_ic_lora_strategy_from_preset()`.

#### Sampling with Tiled VAE

The prompt file format (`--sample_prompts`) — including guidance scale, negative prompt, and per-prompt inference parameters — is documented in the [Sampling During Training guide](./sampling_during_training.md). LTX-2 extends this with `--v <path>` (IC-LoRA reference) and `--ra <path>` (audio-reference IC-LoRA) prompt prefixes.

| Argument | Default | Description |
|----------|---------|-------------|
| `--height` | 512 | Sample output height (pixels) |
| `--width` | 768 | Sample output width (pixels) |
| `--sample_num_frames` | 45 | Number of frames for sample video generation |
| `--sample_with_offloading` | off | Offload DiT to CPU between sampling prompts to save VRAM |
| `--sample_tiled_vae` | off | Enable tiled VAE decoding during sampling to reduce VRAM |
| `--sample_vae_tile_size` | 512 | Spatial tile size (pixels) |
| `--sample_vae_tile_overlap` | 64 | Spatial tile overlap (pixels) |
| `--sample_vae_temporal_tile_size` | 0 | Temporal tile size in frames (0 = disabled) |
| `--sample_vae_temporal_tile_overlap` | 8 | Temporal tile overlap (frames) |
| `--sample_merge_audio` | off | Merge generated audio into the output `.mp4` |
| `--sample_audio_only` | off | Generate audio-only preview outputs |
| `--sample_disable_audio` | off | Disable audio preview generation during sampling |
| `--sample_audio_subprocess` | on | Decode audio in a subprocess to avoid OOM crashes. Use `--no-sample_audio_subprocess` to decode in-process |
| `--sample_disable_flash_attn` | off | Force SDPA instead of FlashAttention during sampling |
| `--sample_i2v_token_timestep_mask` | on | Use I2V token timestep masking (conditioned tokens use t=0). Use `--no-sample_i2v_token_timestep_mask` to disable |

#### Precached Sample Prompts
To avoid loading Gemma during training for sample generation, you can precache the prompt embeddings:

1. During text encoder caching, add `--precache_sample_prompts --sample_prompts sampling_prompts.txt` to also cache the sample prompt embeddings.
2. During training, add `--use_precached_sample_prompts` (or `--precache_sample_prompts`) to load embeddings from cache instead of running Gemma.
- `--sample_prompts_cache`: Path to the precached embeddings file. Defaults to `<cache_directory>/ltx2_sample_prompts_cache.pt`.

For IC-LoRA / V2V training, you can also precache the conditioning image latents during latent caching (see [Latent Caching Arguments](#latent-caching-arguments)):
1. During latent caching, add `--precache_sample_latents --sample_prompts sampling_prompts.txt`.
2. During training, add `--use_precached_sample_latents` to load conditioning latents from cache instead of loading the VAE encoder.
- `--sample_latents_cache`: Path to the precached latents file. Defaults to `<cache_directory>/ltx2_sample_latents_cache.pt`.

#### Two-Stage Sampling (WIP)

> [!NOTE]
> This feature is work in progress and disabled by default. Two-stage inference generates at half resolution, then upsamples and refines.

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample_two_stage` | off | Enable two-stage inference during sampling |
| `--spatial_upsampler_path` | — | Path to spatial upsampler model. Required when `--sample_two_stage` is set |
| `--distilled_lora_path` | — | Path to distilled LoRA for stage 2 refinement. Optional |
| `--sample_stage2_steps` | 3 | Number of denoising steps for stage 2 |

#### Checkpoint Output Format

Saved LoRA checkpoints are converted to ComfyUI format by default. Both the original musubi-tuner format and the ComfyUI format are kept. For the standalone conversion utility, see `convert_lora.py`.

| Flag | Behavior |
|------|----------|
| *(default)* | Saves both `*.safetensors` (original) and `*.comfy.safetensors` (ComfyUI). |
| `--no_save_original_lora` | Deletes the original after conversion, keeping only `*.comfy.safetensors`. |
| `--no_convert_to_comfy` | Saves only the original `*.safetensors` (no conversion). |
| `--save_checkpoint_metadata` | Saves a `.json` sidecar file alongside each checkpoint with loss, lr, step, and epoch. |

> **Important:** Training can only be resumed from the **original** (non-comfy) checkpoint format. If you plan to use `--resume`, do not use `--no_save_original_lora`.
> ComfyUI-only LoRA files can still be used for warm-starting via `--network_weights`, `--base_weights`, or `--dim_from_weights`; only full `--resume` requires the original checkpoint plus saved training state.

Checkpoint rotation (`--save_last_n_epochs`) cleans up old ComfyUI checkpoints alongside originals. HuggingFace upload (`--huggingface_repo_id`) uploads both formats by default. Use `--no_save_original_lora` to upload only the ComfyUI checkpoint.

#### Resuming Training

Requires `--save_state` to be enabled. State directories contain optimizer, scheduler, and RNG states. See the [Advanced Configuration guide](./advanced_config.md) for general `--save_state` / `--resume` behavior shared across all architectures.

| Flag | Description |
|------|-------------|
| `--resume <path>` | Resume from a specific state directory |
| `--autoresume` | Automatically resume from the latest state in `output_dir`. Ignored if `--resume` is specified. Starts from scratch if no state is found |
| `--reset_optimizer` | Clear optimizer momentum/variance on resume, keep model weights only |
| `--reset_optimizer_params` | Reset optimizer param groups (lr, weight_decay, etc.) to current CLI values on resume, keep momentum/variance |
| `--reset_dataloader` | Skip mid-epoch batch skip, restart epoch from beginning |

**Changing learning rate on resume:** When you resume from a saved state, the optimizer's learning rate is restored from the checkpoint — any new `--learning_rate` value on the command line is silently ignored. To apply a new learning rate, add `--reset_optimizer_params`. This resets lr, weight_decay, and other optimizer param-group settings to your current CLI values while keeping the accumulated momentum/variance intact.

Mid-epoch checkpoints record `step_in_epoch` in `resume_metadata.json`. On resume, already-processed batches are skipped to keep global step consistent. `--reset_dataloader` disables this.

The moving average loss is saved in state checkpoints and restored on resume.

---

## Merge LoRA into Base Model

**Script:** `ltx2_merge_lora_to_model.py`

```bash
python ltx2_merge_lora_to_model.py ^
  --dit base_model.safetensors ^
  --lora_weight lora.safetensors ^
  --save_merged_model merged_model.safetensors
```

### Arguments
- `--dit`: LTX-2 base model checkpoint (required).
- `--lora_weight`: One or more LoRA paths to merge sequentially (required).
- `--lora_multiplier`: Per-LoRA multipliers (default: all 1.0).
- `--save_merged_model`: Output merged model path (required).
- `--device cpu|cuda`: Device for merge computation (default: cuda). Pass `--device cpu` to run on system RAM if you don't have enough VRAM.
- `--audio_video`: Load as audio-video model (for LTXAV checkpoints).

### Notes
- The output contains only transformer weights (VAE, vocoder, and text encoder are loaded separately by training/inference scripts).
- Original checkpoint metadata is preserved, so the merged file is directly usable with `--ltx2_checkpoint`.
- FP8 base models cannot be merged directly — merge into the bf16 base, then use `--fp8_base` at training time for on-the-fly quantization.

---

## Merge LTX-2 LoRAs

Use the dedicated LTX-2 LoRA merger to combine multiple LoRA files into a single LoRA checkpoint.

**Script:** `ltx2_merge_lora.py`

### Example Command (Windows)
```bash
python ltx2_merge_lora.py ^
  --lora_weight path/to/lora_A.safetensors path/to/lora_B.safetensors ^
  --lora_multiplier 1.0 1.0 ^
  --save_merged_lora path/to/merged_lora.safetensors
```

### LoRA Merge Arguments
- `--lora_weight`: Input LoRA paths to merge in order (required).
- `--lora_multiplier`: Per-LoRA multipliers aligned with `--lora_weight`. Use one value to apply the same multiplier to all inputs.
- `--save_merged_lora`: Output merged LoRA path (required).
- `--merge_method concat|orthogonal`: Merge method (default: `concat`). `concat` keeps all ranks by concatenation. `orthogonal` uses SVD refactorization to merge exactly 2 LoRAs with orthogonal projection.
- `--orthogonal_k_fraction`: Fraction of top singular directions projected out bilaterally before combining (default: `0.5`, range `[0, 1]`). Only used with `--merge_method orthogonal`.
- `--orthogonal_rank_mode sum|max|min`: Target rank mode for orthogonal merge (default: `sum`).
- `--dtype auto|float32|float16|bfloat16`: Output tensor dtype. `auto` promotes from input dtypes.
- `--emit_alpha`: Force writing `<module>.alpha` keys in output.

### Notes
- This merger is intended for LTX-2 LoRA formats used in this repo, including Comfy-style `lora_A/lora_B` weights.
- It handles different ranks and partial module overlap across input LoRAs.
- Orthogonal merge requires exactly 2 input LoRAs.

---

## Dataset Configuration

The dataset config is a TOML file with `[general]` defaults and `[[datasets]]` entries. Common options shared across all musubi-tuner architectures — including `frame_extraction` modes, JSONL metadata format, control image support, and resolution bucketing — are documented in the [Dataset Configuration guide](./dataset_config.md). The options below are LTX-2-specific or supplement upstream defaults.

### Image Dataset Notes

Image datasets use the common image schema from [Dataset Configuration](./dataset_config.md), including
`image_directory` or `image_jsonl_file`. For LTX-2 IC-LoRA with image datasets, use `reference_directory`
and `reference_cache_directory`. Internally this is normalized onto the shared image control path, so
other non-IC image workflows can continue using `control_directory`.

### Video Dataset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `video_directory` | string | — | Path to video directory |
| `video_jsonl_file` | string | — | Path to JSONL metadata file |
| `resolution` | int or [int, int] | [960, 544] | Target resolution |
| `target_frames` | [int] | [1] | List of target frame counts |
| `frame_extraction` | string | `"head"` | Frame extraction mode |
| `max_frames` | int | 129 | Maximum number of frames |
| `source_fps` | float | auto-detected | Source video FPS. Auto-detected from video container metadata when not set. Use this to override auto-detection. |
| `target_fps` | float | 25.0 | Target training FPS. Frames are resampled to this rate. When audio is present and the source video has a different FPS, the audio waveform is automatically time-stretched (pitch-preserving) to match the target video duration. |
| `batch_size` | int | 1 | Batch size |
| `num_repeats` | int | 1 | Dataset repetitions |
| `enable_bucket` | bool | false | Enable resolution bucketing |
| `bucket_no_upscale` | bool | false | Prevent upscaling when bucketing (only downscale to fit) |
| `cache_directory` | string | — | Latent cache output directory |
| `reference_directory` | string | — | Reference images/videos for IC-LoRA (matched by filename) |
| `reference_cache_directory` | string | — | Output directory for cached reference latents (IC-LoRA) |
| `reference_audio_directory` | string | — | Reference audio files for audio-reference IC-LoRA (matched by filename stem) |
| `reference_audio_cache_directory` | string | — | Output directory for cached reference audio latents |
| `separate_audio_buckets` | bool | false | Keep audio/non-audio items in separate batches |

### Audio Dataset Options

Audio-only datasets use `audio_directory` instead of `video_directory`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `audio_directory` | string | — | Path to audio file directory |
| `audio_jsonl_file` | string | — | Path to JSONL metadata file |
| `audio_bucket_strategy` | string | `"pad"` | `"pad"` (round-to-nearest, pad + mask) or `"truncate"` (floor, clip to bucket length) |
| `audio_bucket_interval` | float | 2.0 | Bucket step size in seconds |
| `batch_size` | int | 1 | Batch size |
| `num_repeats` | int | 1 | Dataset repetitions |
| `cache_directory` | string | — | Latent cache output directory |

### Example TOML

```toml
[general]
resolution = [768, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
cache_directory = "cache"

[[datasets]]
video_directory = "videos"
target_frames = [1, 17, 33, 49]
target_fps = 25    # optional, defaults to 25
```

### Frame Rate (FPS) Handling

During latent caching, the source FPS is **auto-detected** from each video's container metadata and frames are resampled to `target_fps` (default: 25). The model receives video at the configured temporal rate regardless of the source material.

#### How It Works

1. For each video file, the source FPS is read from the container metadata (`average_rate` or `base_rate`).
2. If `abs(ceil(source_fps) - target_fps) > 1`, frames are resampled (dropped) to match `target_fps`.
3. If the difference is within this threshold (e.g., 23.976 → ceil=24 vs target 25, diff=1), no resampling is done — this avoids spurious frame drops from NTSC rounding (23.976, 29.97, 59.94, etc.).
4. If audio is present (`--ltx2_mode av`), the audio waveform is automatically time-stretched (pitch-preserving) to match the resampled video duration.

#### Common Scenarios

**Default — no FPS config needed:**
```toml
[[datasets]]
video_directory = "videos"
target_frames = [1, 17, 33, 49]
# source_fps: auto-detected per video
# target_fps: defaults to 25
```
A 60fps video is resampled to 25fps. A 30fps video is resampled to 25fps. A 25fps video is passed through as-is. Mixed-FPS datasets work correctly — each video is resampled independently.

**Training at a non-standard frame rate (e.g., 60fps):**
```toml
[[datasets]]
video_directory = "videos_60fps"
target_frames = [1, 17, 33, 49]
target_fps = 60
```
Set `target_fps` to the desired training rate. Videos at 60fps (or 59.94fps) pass through without resampling. Videos at other frame rates are resampled to 60fps.

**Overriding auto-detection (e.g., variable frame rate videos):**
```toml
[[datasets]]
video_directory = "videos"
target_frames = [1, 17, 33, 49]
source_fps = 30
```
If auto-detection gives wrong results (common with variable frame rate / VFR recordings from phones), set `source_fps` explicitly. This applies to **all** videos in that dataset entry, so group videos by FPS into separate `[[datasets]]` blocks if needed.

**Image directories:**

Image directories have no FPS metadata. No resampling is applied — all images are loaded as individual frames regardless of `target_fps`.

#### Log Messages

During latent caching, log messages confirm what's happening for each video:
```
Auto-detected source FPS: 60.00 for my_video.mp4
Resampling my_video.mp4: 60.00 FPS -> 25.00 FPS
```
If you see **no** "Resampling" line for a video, it means source and target FPS were close enough (within 1 FPS after rounding up the source) and all frames were kept as-is. If you see unexpected frame counts in your cached latents, check these log lines first.

#### Quick Reference

| Your situation | What to set | What happens |
|---|---|---|
| Mixed FPS dataset, want 25fps training | Nothing (defaults work) | Each video auto-detected, resampled to 25fps |
| All videos are 25fps | Nothing | Auto-detected as 25fps, no resampling |
| All videos are 60fps, want 60fps training | `target_fps = 60` | Auto-detected as 60fps, no resampling |
| All videos are 60fps, want 25fps training | Nothing | Auto-detected as 60fps, resampled to 25fps |
| VFR videos with wrong detection | `source_fps = 30` (your actual FPS) | Overrides auto-detection |
| Image directory | Nothing | No FPS concept, all images loaded |

---

## Validation Datasets

> [!NOTE]
> Validation datasets are a fork extension — they are not available in upstream musubi-tuner.

You can configure a separate validation dataset to track validation loss (`val_loss`) during training. This helps detect overfitting and compare training runs. Validation datasets use **exactly the same schema** as training datasets — any format that works for `[[datasets]]` works for `[[validation_datasets]]`.

### Configuration

Add a `[[validation_datasets]]` section to your existing TOML config file:

```toml
[general]
resolution = [768, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

# Training data
[[datasets]]
video_directory = "videos/train"
cache_directory = "cache/train"
target_frames = [1, 17, 33, 49]

# Validation data
[[validation_datasets]]
video_directory = "videos/val"
cache_directory = "cache/val"
target_frames = [1, 17, 33, 49]
```

The `cache_directory` for validation must be different from the training cache directory.

### Caching

Validation datasets are automatically picked up by the caching scripts — no extra flags needed. Run the same caching commands you use for training:

```bash
python ltx2_cache_latents.py --dataset_config dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors --ltx2_mode av ...
python ltx2_cache_text_encoder_outputs.py --dataset_config dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors --ltx2_mode av ...
```

Both scripts detect the `[[validation_datasets]]` section and cache latents/text embeddings for validation data alongside training data.

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--validate_every_n_steps` | int | None | Run validation every N training steps |
| `--validate_every_n_epochs` | int | None | Run validation every N epochs |

At least one of these must be set for validation to run. If neither is set, validation is skipped even if `[[validation_datasets]]` is configured.

### Example

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --dataset_config dataset.toml ^
  --validate_every_n_steps 100 ^
  ... (other training args)
```

### How It Works

1. A separate validation dataloader is created with `batch_size=1` and `shuffle=False` (deterministic order).
2. At the configured interval, the model switches to eval mode and runs inference on all validation samples with `torch.no_grad()`.
3. The average MSE loss is computed and logged as `val_loss` to TensorBoard/WandB.
4. The model is restored to training mode and training continues.

### Tips

- **Keep validation sets small.** Aim for 5-20% of your main dataset size. Validation runs on every sample each time, so 10-50 clips is usually enough. Large validation sets slow down training.
- **Use held-out data.** Validation data should be different from the training set for meaningful overfitting detection. In extreme cases, using a small subset of the training data is acceptable — it will still help catch divergence, but won't reliably detect overfitting.
- **Monitor the gap.** If `val_loss` starts increasing while training loss keeps decreasing, you're overfitting — consider stopping or reducing the learning rate.
- **Same preprocessing.** Validation data goes through the same frame extraction, FPS resampling, and resolution bucketing as training data.

---

## Directory Structure

### Raw Dataset Layout (Example)
```
dataset_root/
  videos/
    000001.mp4
    000001.txt       # caption
    000002.mp4
    000002.txt
```

### Cache Directory Layout (After Caching)
```
cache_directory/
  000001_1024x0576_ltx2.safetensors       # video latents
  000001_ltx2_te.safetensors              # text encoder outputs
  000001_ltx2_audio.safetensors           # audio latents (av mode only)
  000001_1024x0576_ltx2_dino.safetensors  # DINOv2 features (CREPA dino mode only)

reference_cache_directory/                  # IC-LoRA only
  000001_1024x0576_ltx2.safetensors       # reference latents
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Missing cache keys during training | Caching incomplete | Run both `ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py` |
| FlashAttention varlen mask-length mismatch (for example `expects mask length 1920, got 1024`) after checkpoint switch | Stale text cache from a different checkpoint/version/mode | Re-run `ltx2_cache_text_encoder_outputs.py` with the same `--ltx2_checkpoint` and `--ltx2_mode` as training. Remove old `*_ltx2_te.safetensors` if needed. |
| Samples become progressively noisier/degraded when training with `--flash_attn` | FlashAttention install/runtime mismatch (CUDA/PyTorch/flash-attn build) | Switch to `--sdpa` to confirm baseline stability. If SDPA is stable, reinstall FlashAttention for your exact CUDA + PyTorch versions and retry. |
| Training fails after changing `--ltx2_checkpoint` even though args look correct | Reused latent/text caches generated from a different checkpoint | Re-run both caches (`ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py`) and regenerate `--dataset_manifest` before training. |
| OOM appears after removing `--fp8_base` while using an FP8 checkpoint | Base model no longer uses FP8 loading path, so VRAM increases sharply | Keep `--fp8_base` enabled for FP8 checkpoints (typically with `--mixed_precision bf16`) |
| Error when combining `--fp8_scaled` with an FP8 checkpoint (old behavior) | Checkpoint has FP8 weights without `weight_scale` keys (non-standard format) | Use a standard bf16 checkpoint, or a properly exported FP8 checkpoint that includes scale tensors |
| Missing `*_ltx2_audio.safetensors` | Audio caching skipped | Re-run latent caching with `--ltx2_mode av` |
| Gemma connector weights missing | Incorrect checkpoint | Ensure `--ltx2_checkpoint` (or `--ltx2_text_encoder_checkpoint`) contains Gemma connector weights |
| Gemma OOM | Model too large | Use `--gemma_load_in_8bit` or `--gemma_load_in_4bit` with `--device cuda`, or use `--gemma_safetensors` with an FP8 file |
| Audio caching fails | torchaudio missing | Install torchaudio before running `ltx2_cache_latents.py` |
| Sampling OOM | VAE decode too large | Enable `--sample_tiled_vae` or reduce `--sample_vae_temporal_tile_size` |
| Crash with block swap (esp. RTX 5090) | `--use_pinned_memory_for_block_swap` bug | Remove `--use_pinned_memory_for_block_swap` from training arguments |
| `stack expects each tensor to be equal size` during AV training | Mixed audio/non-audio videos in the same batch — text embeddings are 2×`caption_channels` for AV items vs 1×`caption_channels` for video-only (e.g., 7680 vs 3840 for LTX-2.3), and `torch.stack` fails | Add `--separate_audio_buckets` to training args. Required when your dataset mixes videos with and without audio at `batch_size > 1`. At `batch_size=1` it has no effect. |
| Wrong frame count in cached latents | Auto-detected FPS incorrect (e.g., VFR video) | Set `source_fps` explicitly in TOML config to override auto-detection |
| Too few frames from high-FPS video | FPS resampling working correctly (e.g., 60fps→25fps = 42% of frames) | Expected behavior. Set `target_fps = 60` if you want to keep all frames |
| Audio/video out of sync after caching | Source FPS mismatch causing wrong time-stretch | Check "Auto-detected source FPS" log line; set `source_fps` explicitly if wrong |
| Voice/audio learning slow when mixing images with videos in AV mode | Image batches produce zero audio training signal — audio branch is skipped entirely. Dilutes audio learning proportionally to image step fraction | Use video-only datasets for AV training when voice quality matters |
| No audio during sampling in video training mode | `ltx2_mode` is set to `v`/`video` | Expected behavior. Train in AV mode (`--ltx2_mode av` or `audio`) to generate audio during sampling |
| Cannot resume training from checkpoint | Using a `*.comfy.safetensors` checkpoint with `--resume` | Training can only be resumed from the **original** (non-comfy) LoRA format. Use the `*.safetensors` file without the `.comfy` extension. If you used `--no_save_original_lora`, you must retrain from scratch. |
| CUDA errors or crashes on RTX 5090 / 50xx GPUs | CUDA 12.6 (`cu126`) not supported on Windows for Blackwell GPUs | Use CUDA 12.8: `pip install torch==2.8.0 ... --index-url https://download.pytorch.org/whl/cu128`. See [CUDA Version](#cuda-version) |
| `ValueError: Gemma safetensors is missing required language-model tensors` with `missing_buffers` mentioning `full_attention_inv_freq` or `sliding_attention_inv_freq` | `transformers>=5.0` renamed Gemma3 rotary embedding buffers (`rotary_emb.inv_freq` → `rotary_emb.full_attention_inv_freq` / `sliding_attention_inv_freq`). The derivable-buffer suffix check expects `.inv_freq` and does not match the new `_inv_freq` suffix. The safetensors file is correct — rotary buffers are non-persistent and computed from config at init time. | `pip install transformers==4.56.1` (pinned in `pyproject.toml`), or reinstall all deps with `pip install -e .` |
| Audio quality degrades after training video/image LoRA on an AV checkpoint | Default `t2v` preset creates LoRA weights for audio and cross-modal attention layers. With no audio training data, those weights are initialized but receive no meaningful gradient signal — applying the LoRA overwrites audio layers with near-zero deltas that disrupt the base model's audio representations. | Use a `video_*` preset (`--lora_target_preset video_sa`, `video_sa_ff`, or `video_sa_ca_ff`) to restrict LoRA to video-branch modules only. Audio layers remain frozen and unmodified. See [LoRA Targets](#lora-targets). |
| `loss_a` too low but `loss_v` still high (audio overfitting) | Audio latent space converges faster than video; audio gradients dominate shared weights | Lower `--audio_loss_weight` (e.g., 0.3), or use `--audio_loss_balance_mode ema_mag` to auto-dampen audio when it exceeds `target_ratio × video_loss`. Reduce audio learning rate with `--audio_lr 1e-6` or fine-grained `--lr_args audio_attn=1e-6 audio_ff=1e-6`. Disable `--audio_dop` / `--audio_silence_regularizer` if active — they add more audio signal. |
| `loss_a` absent or not dropping in mixed dataset (audio starvation) | Audio batches too rare — non-audio steps outnumber audio steps, audio branch gets insufficient supervision | Increase `num_repeats` on audio datasets (target 30-50% audio steps). Add `--audio_loss_balance_mode inv_freq` to auto-boost audio weight. Use `--audio_dop` or `--audio_silence_regularizer` to provide audio signal on non-audio steps. Check caching summary for `failed > 0`. |

### Audio/Voice Training with Mixed Datasets

Audio overfits faster than video due to lower-dimensional latent space and simpler temporal structure. In mixed datasets, audio batches are also a minority — non-audio steps update shared weights without audio supervision, causing audio drift. The tools below address both problems.

**1. Oversample audio items** — set `num_repeats` so audio batches are 30-50% of steps:
```toml
[[datasets]]
video_directory = "audio_video_clips"
num_repeats = 5
```

**2. Lower audio learning rate** — audio converges faster, so reduce its LR relative to video. Use `--audio_lr` for a blanket reduction or `--lr_args` for per-module control. Joint AV training typically needs a lower base LR than video-only (UniAVGen uses 4x lower LR for joint vs single-modality training).
```
--learning_rate 1e-4 --audio_lr 3e-5
```

**3. Lower audio LoRA rank** — audio needs less adaptation capacity. Use `--audio_dim` to set a smaller rank for audio modules while keeping video rank higher:
```
--network_dim 32 --audio_dim 8 --audio_alpha 8
```

**4. `--audio_dop`** — preserves base model audio predictions on non-audio steps. Runs LoRA OFF/ON forwards on silence audio latents, MSE on audio branch only. +2 fwd / +1 bwd per non-audio step, zero cost on audio batches. Logged as `loss/audio_dop`. Mutually exclusive with `--audio_silence_regularizer`.
```
--audio_dop --audio_dop_args multiplier=0.5
```

**5. `--audio_silence_regularizer`** — converts non-audio batches to audio batches with silence target. Cheaper than DOP (+0 extra forwards) but silence is an approximation.
```
--audio_silence_regularizer --audio_silence_regularizer_weight 0.5
```

**6. Loss balancing** (`--audio_loss_balance_mode`) — four modes for dynamic audio loss weight adjustment:
- `inv_freq` — scales audio weight by inverse of audio-batch frequency EMA. Compensates when audio batches are rare:
  ```
  --audio_loss_balance_mode inv_freq --audio_loss_balance_min 0.05 --audio_loss_balance_max 4.0
  ```
- `ema_mag` — tracks audio/video loss magnitude EMAs and scales audio weight to match a target ratio (default 0.33). Bidirectional — dampens audio when too high, boosts when too low:
  ```
  --audio_loss_balance_mode ema_mag --audio_loss_balance_target_ratio 0.33
  ```
- `uncertainty` — two learnable log-variance scalars optimized jointly with LoRA weights (Kendall et al., CVPR 2018). No manual weight tuning required — scalars are learned via backpropagation:
  ```
  --audio_loss_balance_mode uncertainty
  ```
- `ogm_ge` — attenuates the lower-loss / faster-learning modality on each AV step. Optional gradient noise is available but disabled by default:
  ```
  --audio_loss_balance_mode ogm_ge --ogm_ge_alpha 0.3
  ```

**7. Independent modality dropout** — drop video or audio text conditioning independently per sample. Serves as both anti-dominance regularization and CFG training:
```
--video_caption_dropout_rate 0.1 --audio_caption_dropout_rate 0.15
```

**8. Modality freezing** — auto-freezes the dominant modality's LoRA when loss ratio crosses a threshold (G2D, 2025). Lets the under-performing modality train without gradient interference:
```
--modality_freeze_check_interval 500 --modality_freeze_ratio_threshold 0.5
```

**9. Self-Flow audio alignment** — anchors audio hidden states to the base model's representations via cosine similarity loss, preventing audio feature drift during LoRA adaptation:
```
--self_flow --self_flow_args lambda_self_flow=0.1 lambda_audio=0.1 teacher_mode=base
```

**10. Cross-Task Synergy** — auxiliary losses with one modality clean (timestep=0) provide stable cross-modal alignment targets ([Harmony, 2025](https://arxiv.org/abs/2511.21579)). Adds two extra forward passes per AV batch:
```
--cts_lambda_video_driven 0.3 --cts_lambda_audio_driven 0.1
```

**11. Diagnostics** — per-modality gradient norms (`grad_norm/video`, `grad_norm/audio`, `grad_norm/audio_video_ratio`) are logged automatically in AV mode. A ratio deviating >3x from its initial value indicates modality imbalance.

**12. Per-group warmup overrides** — keep the same scheduler family but stretch warmup differently for audio and video LR groups:
```
--lr_group_warmup_args audio=500 video=1500
```

- If `failed > 0` in latent caching summary, audio extraction is broken for those items
- After mode switch (video→AV), re-run both latent and text encoder caching without `--skip_existing`
- `loss_a` dropping = audio learning; absent/zero = no audio batches forming; degrades over time = forgetting

### Technical Notes

- **Float32 AdaLN**: The transformer applies Adaptive Layer Norm (AdaLN) shift/scale operations in float32, then casts back to the working dtype. This prevents overflow that can occur when bf16 scale values multiply bf16 hidden states. The fix is always active and requires no flags.
- **Loss dtype**: The LTX-2 training path computes the task loss (MSE, L1, Huber) in `trainer.dit_dtype` (typically bf16 with `--mixed_precision bf16`). Internal regularization losses (motion preservation, CREPA, Self-Flow) always use MSE and are unaffected by `--loss_type`.

For additional troubleshooting resources, see the [official LTX-2 documentation hub](https://docs.ltx.video/open-source-model/getting-started/overview), the [Banodoco Discord](https://discord.gg/banodoco) community, and the [awesome-ltx2](https://github.com/wildminder/awesome-ltx2) curated resource list.

---

## 4. Slider LoRA Training

> Slider LoRA training is based on the [ai-toolkit](https://github.com/ostris/ai-toolkit) implementation by ostris, adapted for LTX-2.

Slider LoRAs learn a controllable direction in model output space (e.g., "detailed" vs "blurry"). At inference, you scale the LoRA multiplier to control the effect strength and direction: `+1.0` enhances, `-1.0` erases, `0.0` is the base model, and values like `+2.0` or `-0.5` work too.

**Script:** `ltx2_train_slider.py`

Three modes are available:

| Mode | Input | Use Case |
|------|-------|----------|
| `text` | Prompt pairs only (no dataset) | Sliders from text prompt pairs, no images needed |
| `reference` | Pre-cached latent pairs | Sliders from paired positive/negative image, video, or audio samples |
| IC-slider (`mode = "ic_reference"`) | Paired target caches + shared reference caches | Slider training under shared `v2v` reference conditioning |

### 4a. Text-Only Mode

Learns a slider direction from positive/negative prompt pairs. No images or dataset config needed.

#### Slider Config (`ltx2_slider.toml`)

```toml
mode = "text"
guidance_strength = 1.0
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]

[[targets]]
positive = "extremely detailed, sharp, high resolution, 8k"
negative = "blurry, out of focus, low quality, soft"
target_class = ""      # empty = affect all content
weight = 1.0
```

- `guidance_strength`: Scales the directional offset applied to targets. Higher values = stronger direction signal but may overshoot.
- `target_class`: The conditioning prompt used during training passes. Empty string means the slider affects all content regardless of prompt. Set to e.g. `"a portrait"` to restrict the slider's effect to a specific subject.
- `weight`: Per-target loss weight. Controls relative emphasis when training multiple directions simultaneously.
- `sample_slider_range`: Multiplier values used for preview samples during training.

Multiple `[[targets]]` blocks can be defined to train several directions at once (e.g., detail + lighting).

#### Example Command (Text-Only)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_slider.py ^
  --mixed_precision bf16 ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --gemma_root /path/to/gemma ^
  --gemma_load_in_8bit ^
  --fp8_base --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap 10 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --lora_target_preset t2v ^
  --learning_rate 1e-4 ^
  --optimizer_type AdamW8bit ^
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 ^
  --max_train_steps 500 ^
  --output_dir output --output_name detail_slider ^
  --slider_config ltx2_slider.toml ^
  --latent_frames 1 ^
  --latent_height 512 --latent_width 768
```

#### Text-Only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--slider_config` | (required) | Path to slider TOML config file |
| `--latent_frames` | 1 | Number of latent frames (1=image, >1=video) |
| `--latent_height` | 512 | Pixel height for synthetic latents |
| `--latent_width` | 768 | Pixel width for synthetic latents |
| `--guidance_strength` | (from TOML) | Override `guidance_strength` from config |
| `--sample_slider_range` | (from TOML) | Override as comma-separated values, e.g. `"-2,-1,0,1,2"` |

All standard training arguments (`--fp8_base`, `--blocks_to_swap`, `--gradient_checkpointing`, etc.) work the same as regular training. `--dataset_config` is not needed for text-only mode.

### 4b. Reference Mode

Learns a slider direction from paired positive/negative image, video, or audio examples. Requires pre-cached latents.

#### Step 1: Prepare Paired Data

Create two directories with matching filenames — one with positive-attribute images, one with negative:

```
positive_images/        negative_images/
  img_001.png             img_001.png      # same subject, different attribute
  img_002.png             img_002.png
  img_003.png             img_003.png
```

Each positive image must have a corresponding negative image with the same filename. The images should depict the same subject but differ in the target attribute (e.g., smiling vs neutral face, detailed vs blurry).

For image-based sliders, cache these paired images normally and keep `reference_modality = "video"` (the visual latent path covers both single-frame images and multi-frame videos).

#### Step 2: Cache Latents and Text

Create a dataset config for each directory and run the caching scripts. Both directories can share the same text captions (since the direction comes from the images, not the text).

```bash
# Cache positive latents
python ltx2_cache_latents.py --dataset_config positive_dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors

# Cache negative latents
python ltx2_cache_latents.py --dataset_config negative_dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors

# Cache text (once, for either directory — text is shared)
python ltx2_cache_text_encoder_outputs.py --dataset_config positive_dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors --gemma_root /path/to/gemma
```

#### Step 3: Configure and Train

##### Slider Config (`ltx2_slider_reference.toml`)

```toml
mode = "reference"
reference_modality = "video"            # "video" or "audio"
pos_cache_dir = "path/to/positive/cache"
neg_cache_dir = "path/to/negative/cache"
text_cache_dir = "path/to/positive/cache"   # can be same as pos (text is shared)
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

- `reference_modality`: `video` for image/video latent pairs, `audio` for paired audio latents
- `pos_cache_dir`: Directory containing cached positive latents (output of `ltx2_cache_latents.py`)
- `neg_cache_dir`: Directory containing cached negative latents
- `text_cache_dir`: Directory containing cached text encoder outputs. Defaults to `pos_cache_dir` if omitted.

Video pairs are matched by filename: for each `{name}_{W}x{H}_ltx2.safetensors` in the positive directory, a matching file must exist in the negative directory.

Audio pairs are matched by filename: for each `{name}_ltx2_audio.safetensors` in the positive directory, a matching audio file must exist in the negative directory, and both directories must also contain the companion audio-only virtual geometry cache `{name}_{W}x{H}_ltx2.safetensors`. Unmatched files are skipped with a warning.

##### Example Command (Reference)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_slider.py ^
  --mixed_precision bf16 ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --fp8_base --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap 10 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --lora_target_preset t2v ^
  --learning_rate 1e-4 ^
  --optimizer_type AdamW8bit ^
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 ^
  --max_train_steps 500 ^
  --output_dir output --output_name smile_slider ^
  --slider_config ltx2_slider_reference.toml
```

Note: `--gemma_root` is not needed for reference mode (text embeddings are loaded from cache). `--dataset_config`, `--latent_frames/height/width` are also not used.

For video reference sliders, `--ltx2_first_frame_conditioning_p` also works here. When enabled on multi-frame samples, the trainer anchors frame 0 as conditioning-only and excludes it from the loss, which is useful when positive/negative pairs share the same start frame and differ mainly in motion. It has no effect for text-only sliders or single-frame reference samples.

##### Audio Reference Sliders

For paired audio sliders, set `reference_modality = "audio"` and train in audio-only mode:

```toml
mode = "reference"
reference_modality = "audio"
pos_cache_dir = "path/to/positive/audio_cache"
neg_cache_dir = "path/to/negative/audio_cache"
text_cache_dir = "path/to/positive/audio_cache"
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_slider.py ^
  --mixed_precision bf16 ^
  --ltx2_checkpoint /path/to/ltxav-2.safetensors ^
  --ltx2_mode audio ^
  --fp8_base --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap 10 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --lora_target_preset audio ^
  --learning_rate 1e-4 ^
  --optimizer_type AdamW8bit ^
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 ^
  --max_train_steps 500 ^
  --sample_audio_only ^
  --output_dir output --output_name audio_energy_slider ^
  --slider_config ltx2_slider_audio_reference.toml
```

Notes:
- Audio reference sliders are an MVP path built for `--ltx2_mode audio`.
- The trainer uses the same paired positive/negative audio latents plus shared text cache, and masks loss with cached `audio_lengths`.
- `--lora_target_preset audio` is recommended; if omitted, the slider trainer selects it automatically for audio reference sliders.
- `--ltx2_first_frame_conditioning_p` has no effect for audio sliders.

### 4c. IC-slider
Trains a slider from paired positive/negative target latents under a shared cached visual reference. Internally this mode reuses the existing `v2v` IC path.

#### Slider Config (`ltx2_slider_ic_reference.toml`)

```toml
mode = "ic_reference"
reference_modality = "video"
pos_cache_dir = "path/to/positive/cache"
neg_cache_dir = "path/to/negative/cache"
text_cache_dir = "path/to/text/cache"
reference_cache_dir = "path/to/reference/cache"
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

- `pos_cache_dir`: Directory containing cached positive target latents.
- `neg_cache_dir`: Directory containing cached negative target latents.
- `text_cache_dir`: Directory containing cached text encoder outputs matched by basename.
- `reference_cache_dir`: Directory containing cached reference-video latents matched by basename.

Current restrictions:
- `reference_modality = "video"` only
- `--ltx2_mode video` only
- `--ic_lora_strategy` resolves to `v2v`
- if `--lora_target_preset` is omitted, the trainer selects `v2v`

Files are matched by basename. For each `{name}_{W}x{H}_ltx2.safetensors` in `pos_cache_dir`, the trainer expects:
- a matching negative latent file in `neg_cache_dir`
- a matching text cache `{name}_ltx2_te.safetensors` in `text_cache_dir`
- a matching reference latent file in `reference_cache_dir`

##### Example Command (IC-Aware Reference)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_slider.py ^
  --mixed_precision bf16 ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --ltx2_mode video ^
  --fp8_base --fp8_scaled ^
  --gradient_checkpointing ^
  --blocks_to_swap 10 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --lora_target_preset v2v ^
  --learning_rate 1e-4 ^
  --optimizer_type AdamW8bit ^
  --lr_scheduler constant_with_warmup --lr_warmup_steps 20 ^
  --max_train_steps 500 ^
  --output_dir output --output_name identity_smile_slider ^
  --slider_config ltx2_slider_ic_reference.toml
```

Additional notes:
- `--ltx2_first_frame_conditioning_p` still applies on the target side
- AV and audio IC-slider variants are not implemented

### Slider Tips

- **Start small**: `--network_dim 8` or `16` with `--max_train_steps 200-500` is usually sufficient.
- **Monitor loss**: Loss should decrease steadily. If it diverges, reduce `--learning_rate`.
- **Preview samples**: Add `--sample_prompts sampling_prompts.txt --sample_every_n_steps 50` to generate previews at each slider strength during training. Requires `--gemma_root` for text encoding. For audio sliders, also use `--sample_audio_only`.
- **Guidance strength**: For text-only mode, the default is `1.0`. Values of `2.0-3.0` increase direction strength but may reduce convergence stability.
- **Multiple targets**: Text-only mode supports multiple `[[targets]]` blocks. Each step randomly selects one target, so all directions get trained evenly.
- **Inference**: Use the trained LoRA with any multiplier value. Positive multipliers enhance the positive attribute, negative multipliers enhance the negative attribute. Values beyond `[-1, +1]` extrapolate the effect.

---

## References

**Musubi Tuner Documentation**
- [Dataset Configuration](./dataset_config.md) — TOML format, `frame_extraction` modes, JSONL metadata, control images, resolution bucketing
- [Sampling During Training](./sampling_during_training.md) — Prompt file format, per-prompt guidance scale, negative prompts, sampling CLI flags
- [Advanced Configuration](./advanced_config.md) — `--config_file` TOML training configuration, `--network_args` format, LoRA+, TensorBoard/WandB logging, PyTorch Dynamo, timestep bucketing, Schedule-Free optimizer
- [Tools](./tools.md) — Post-hoc EMA LoRA merging, image captioning with Qwen2.5-VL
- [LoHa/LoKr](./loha_lokr.md) — Alternative parameter-efficient fine-tuning methods
- [torch.compile](./torch_compile.md) — PyTorch JIT compilation for faster training and inference
- [LyCORIS Algorithm List](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md) and [Guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md) — LoKR, LoHA, LoCoN and other algorithm details (used via `pip install lycoris-lora`)

**Research**
- [ID-LoRA](https://github.com/ID-LoRA/ID-LoRA) — In-context identity LoRA; the audio-reference IC-LoRA implementation in this trainer is based on this approach
- [CREPA (arXiv 2506.09229)](https://arxiv.org/abs/2506.09229) — Cross-frame Representation Alignment; basis for `--crepa dino` mode (DINOv2 teacher from neighboring frames)
- [Self-Flow (arXiv 2603.06507)](https://arxiv.org/abs/2603.06507) — Self-supervised flow matching regularization; basis for `--self_flow`
- [Harmony (arXiv 2511.21579)](https://arxiv.org/abs/2511.21579) — Cross-Task Synergy; basis for `--cts_lambda_video_driven` and `--cts_lambda_audio_driven`

**Official LTX Resources**
- [LTX-2](https://github.com/Lightricks/LTX-2) — Official Lightricks LTX-2/2.3 repository; contains the well-structured `ltx-trainer` and `ltx-pipelines` packages that served as the upstream source and reference for this implementation
- [LTX-Video](https://github.com/Lightricks/LTX-Video) — Official Lightricks model repository (inference, ComfyUI nodes, model weights)
- [LTX Documentation](https://docs.ltx.video/open-source-model/getting-started/overview) — Unified docs hub: open-source model, API reference, ComfyUI integration, LoRA usage, and LTX-2 trainer guide

**Alternative Trainers**
- [ai-toolkit](https://github.com/ostris/ai-toolkit) (ostris) — General diffusion fine-tuning toolkit with LTX-2/2.3 LoRA support; slider LoRA training is based on its implementation
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — ModelScope diffusion synthesis framework with LTX-2/2.3 support
- [SimpleTuner](https://github.com/bghira/SimpleTuner) — Multi-model fine-tuning framework with LTX-2/2.3 support; `--crepa backbone` mode is inspired by its LayerSync regularizer

**Community Resources**
- [awesome-ltx2](https://github.com/wildminder/awesome-ltx2) — Curated list of LTX-2 resources, tools, models, and guides
- [Banodoco Discord](https://discord.gg/SrkBPTzw) — Active AI video generation community; discussions on LTX-2 training, workflows, and research
- [Windows Installation Guide](https://github.com/AkaneTendo25/musubi-tuner/discussions/19) — Windows-specific setup (Python 3.12, CUDA, Flash Attention 2), dependencies, troubleshooting
- [LTX-2 Training Optimizers](https://github.com/AkaneTendo25/musubi-tuner/discussions/21) — Optimizer comparison for LTX-2 training: AdamW, Prodigy, Muon, CAME, and recommended settings
- [LTX-2 Audio Dataset Builder](https://github.com/dorpxam/LTX-2-Audio-Dataset-Builder) — Tool to automate audio dataset creation: transforms raw audio into clean, captioned segments optimized for LTX-2 audio-only training

**Tutorials & Guides**
- [LTX-2 LoRA Training Complete Guide](https://apatero.com/blog/ltx-2-lora-training-fine-tuning-complete-guide-2025) (Apatero) — Dataset preparation, training configuration, and LoRA deployment walkthrough
- [How to Train a LTX-2 Character LoRA](https://ghost.oxen.ai/how-to-train-a-ltx-2-character-lora-with-oxen-ai/) (Oxen.ai) — Character-consistency LoRA training with dataset prep tips for audio clips

**Cloud Platforms**
- [fal.ai LTX-2 Trainer](https://fal.ai/models/fal-ai/ltx2-video-trainer) — Cloud-based LTX-2 LoRA training via API (~$0.005/step)
- [WaveSpeedAI LTX-2](https://wavespeed.ai/landing/ltx2) — Hosted LTX-2 inference (T2V, I2V, video extend, lipsync)

---

## Setup / Update Script

[`scripts/install.ps1`](https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2-dev/scripts/install.ps1) is the Windows setup and maintenance entry point.

> [!WARNING]
> The dashboard and `scripts/install.ps1` are still early prototypes and work in progress. Their stable behavior is not guaranteed.

It can:

- install or locate prerequisites (`git`, Python, Node.js)
- clone the repository or update an existing checkout
- create or repair the virtual environment
- install or refresh Python dependencies
- build or rebuild the dashboard frontend
- create or recreate dashboard/setup launchers and desktop shortcuts
- launch the dashboard
- switch the target branch with `-Branch`

**Quick start (one-liner):**

```powershell
irm https://raw.githubusercontent.com/AkaneTendo25/musubi-tuner/ltx-2-dev/scripts/install.ps1 | iex
```

This downloads and runs the script with default settings (CUDA 12.8, Python 3.12, `ltx-2-dev` branch). Interactive mode shows the available actions and lets you choose which ones to run.

**With custom parameters** — save the script locally first:

```powershell
irm https://raw.githubusercontent.com/AkaneTendo25/musubi-tuner/ltx-2-dev/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Cuda cu124 -PythonVersion 3.11 -NonInteractive
```

Available parameters: `-InstallRoot`, `-Branch`, `-Cuda` (`cu124`/`cu128`/`cu130`/`cpu`), `-PythonVersion` (`3.10`-`3.13`), `-Port`, `-DashboardHost`, `-NonInteractive`, `-PreflightOnly`.

On success the script writes launchers and desktop shortcuts for the dashboard and the setup tool, and records install state used by later runs.

The script writes a timestamped log to `%TEMP%`. On failure it prints support details at the end of the run.

