# LTX-2 / LTX-2.3

Supports LoRA training for both **LTX-2 (19B)** and **LTX-2.3 (22B)** models with the following training modes: text-to-video, joint audio-video, audio-only, IC-LoRA / video-to-video, and audio-reference IC-LoRA.

Full-parameter fine-tuning for LTX-2.3 is documented in [Appendix: Full-Parameter Fine-Tuning](#appendix-full-parameter-fine-tuning).

### Supported Model Versions
<sub>[↑ contents](#table-of-contents)</sub>

| Version | Parameters | Key Differences |
|---------|-----------|-----------------|
| LTX-2 (19B) | 19B | Single `aggregate_embed`, caption projection inside transformer |
| LTX-2.3 (22B) | 22B | Dual `video_aggregate_embed`/`audio_aggregate_embed`, caption projection moved to feature extractor (`caption_proj_before_connector`), cross-attention AdaLN (`prompt_adaln`), separate audio connector dimensions, BigVGAN v2 vocoder with bandwidth extension |

Version choice for training is controlled by the `--ltx_version` flag (default: `2.3`): pass `--ltx_version 2.0` for the LTX-2 (19B) checkpoint and `--ltx_version 2.3` for the LTX-2.3 (22B) checkpoint. The trainer auto-detects the checkpoint version from metadata and warns on mismatch.

Caching scripts (`ltx2_cache_latents.py`, `ltx2_cache_text_encoder_outputs.py`) also accept `--ltx_version`, but only for sample-prompt precaching defaults (`--precache_sample_latents` / `--precache_sample_prompts`). Dataset cache compatibility is still driven by `--ltx2_checkpoint` and `--ltx2_mode`.

---

## Table of Contents

- [Supported Model Versions](#supported-model-versions)
- [Part I — Getting Started](#part-i--getting-started)
  - [Installation](#installation)
    - [CUDA Version](#cuda-version)
    - [Downloading Required Models](#downloading-required-models)
  - [Quick Start](#quick-start)
  - [Supported Dataset Types](#supported-dataset-types)
  - [Dataset Configuration](#dataset-configuration)
    - [Image Dataset Notes](#image-dataset-notes)
    - [Video Dataset Options](#video-dataset-options)
    - [Audio Dataset Options](#audio-dataset-options)
    - [Masked Loss Datasets](#masked-loss-datasets)
    - [Example TOML](#example-toml)
    - [Frame Rate (FPS) Handling](#frame-rate-fps-handling)
      - [How It Works](#how-it-works)
      - [Common Scenarios](#common-scenarios)
      - [Log Messages](#log-messages)
      - [Quick Reference](#quick-reference)
  - [Caching Latents](#caching-latents)
    - [Latent Caching Command](#latent-caching-command)
    - [Latent Caching Arguments](#latent-caching-arguments)
    - [Latent Cache Output Files](#latent-cache-output-files)
    - [Latent Cache Preview and Verification](#latent-cache-preview-and-verification)
    - [Memory Optimization for Caching](#memory-optimization-for-caching)
  - [Caching Text Encoder Outputs](#caching-text-encoder-outputs)
    - [Text Encoder Caching Command](#text-encoder-caching-command)
    - [Text Encoder Caching Arguments](#text-encoder-caching-arguments)
    - [Text Encoder Output Files](#text-encoder-output-files)
    - [Loading Gemma from a Single Safetensors File](#loading-gemma-from-a-single-safetensors-file)
  - [Training Your First LoRA](#training-your-first-lora)
    - [Choosing Model Version for Training (2.0 vs 2.3)](#choosing-model-version-for-training-20-vs-23)
    - [Standard LoRA Training](#standard-lora-training)
    - [Audio-Only Training](#audio-only-training)
- [Part II — Core Workflows](#part-ii--core-workflows)
  - [Sampling, Checkpoints & Resuming](#sampling-checkpoints--resuming)
    - [Sampling with Tiled VAE](#sampling-with-tiled-vae)
    - [Precached Sample Prompts](#precached-sample-prompts)
    - [Two-Stage Sampling](#two-stage-sampling)
    - [Checkpoint Output Format](#checkpoint-output-format)
    - [Extract LoRA from Full Fine-Tune](#extract-lora-from-full-fine-tune)
    - [Resuming Training](#resuming-training)
  - [Merge LoRA into Base Model](#merge-lora-into-base-model)
    - [Merge-to-Base Arguments](#merge-to-base-arguments)
    - [Merge-to-Base Notes](#merge-to-base-notes)
  - [Merge LTX-2 LoRAs](#merge-ltx-2-loras)
    - [Example Command (Windows)](#example-command-windows)
    - [LoRA Merge Arguments](#lora-merge-arguments)
    - [LoRA Merge Notes](#lora-merge-notes)
  - [Validation Datasets](#validation-datasets)
    - [Configuration](#configuration)
    - [Caching](#caching)
    - [Training Arguments](#training-arguments)
    - [Example](#example)
    - [How It Works](#how-it-works-1)
    - [Tips](#tips)
  - [Directory Structure](#directory-structure)
    - [Raw Dataset Layout (Example)](#raw-dataset-layout-example)
    - [Cache Directory Layout (After Caching)](#cache-directory-layout-after-caching)
  - [Troubleshooting](#troubleshooting)
    - [Mixed Audio-Video Training](#mixed-audio-video-training)
    - [Technical Notes](#technical-notes)
  - [Windows Setup / Update Script](#windows-setup--update-script)
    - [Dashboard Usage](#dashboard-usage)
- [Part III — Advanced](#part-iii--advanced)
  - [Advanced Training](#advanced-training)
    - [Forward Explorative Modeling (XM)](#forward-explorative-modeling-xm)
    - [Optional: Source-Free Training from Cache](#optional-source-free-training-from-cache)
    - [DoRA LoRA Training](#dora-lora-training)
    - [Rank-Stabilized LoRA (rsLoRA)](#rank-stabilized-lora-rslora)
    - [Advanced: LyCORIS/LoKR Training](#advanced-lycorislokr-training)
    - [Memory Optimization](#memory-optimization)
      - [Quantization Options](#quantization-options)
      - [Other Memory Options](#other-memory-options)
    - [Bounded Activation Offload](#bounded-activation-offload)
    - [Blockwise Checkpointing](#blockwise-checkpointing)
    - [Aggressive VRAM Optimization (8-16GB GPUs)](#aggressive-vram-optimization-8-16gb-gpus)
    - [Low Main-RAM Training](#low-main-ram-training)
      - [Reducing the Main-RAM Cost of Block Swap](#reducing-the-main-ram-cost-of-block-swap)
    - [NF4 Quantization](#nf4-quantization)
    - [NVFP4 (FP4 E2M1) Checkpoints](#nvfp4-fp4-e2m1-checkpoints)
    - [int8 Base (Optimum-Quanto)](#int8-base-optimum-quanto)
    - [INT8 ConvRot Base](#int8-convrot-base)
    - [W4A4G4 Training](#w4a4g4-training)
      - [ComfyUI inference export (convrot_w4a4)](#comfyui-inference-export-convrot_w4a4)
      - [NVFP4 container](#nvfp4-container)
    - [Model Version](#model-version)
    - [Audio-Video Support](#audio-video-support)
    - [Loss Function Type](#loss-function-type)
    - [Loss Weighting](#loss-weighting)
    - [Additional Audio Training Flags](#additional-audio-training-flags)
    - [Modality Freezing (G2D)](#modality-freezing-g2d)
    - [Frozen Auxiliary LoRA During Training](#frozen-auxiliary-lora-during-training)
    - [Optimizers](#optimizers)
    - [Per-Module Learning Rates](#per-module-learning-rates)
    - [Per-Module LoRA Rank](#per-module-lora-rank)
    - [Adaptive LoRA Rank](#adaptive-lora-rank)
    - [Per-Module LoRA Dropout](#per-module-lora-dropout)
    - [Standalone Inference](#standalone-inference)
      - [Rendering Trained Conditioning](#rendering-trained-conditioning)
    - [Audio Quality Metrics](#audio-quality-metrics)
    - [Timestep Sampling](#timestep-sampling)
    - [LoRA Targets](#lora-targets)
    - [LoRA Target Estimation (`ltx2_estimate.py`)](#lora-target-estimation-ltx2_estimatepy)
    - [Connector LoRA (`--train_connectors`)](#connector-lora---train_connectors)
  - [Regularization & Objectives](#regularization--objectives)
    - [Preservation & Regularization](#preservation--regularization)
    - [CREPA (Cross-frame Representation Alignment)](#crepa-cross-frame-representation-alignment)
      - [CREPA CLI Flags](#crepa-cli-flags)
      - [CREPA Parameters (`--crepa_args`)](#crepa-parameters---crepa_args)
      - [CREPA Checkpoint & Resume](#crepa-checkpoint--resume)
      - [CREPA Monitoring](#crepa-monitoring)
      - [CREPA Compatibility](#crepa-compatibility)
      - [Caching DINOv2 Features (Dino Mode)](#caching-dinov2-features-dino-mode)
    - [Self-Flow (Self-Supervised Flow Matching)](#self-flow-self-supervised-flow-matching)
      - [CLI Flags](#cli-flags)
      - [Self-Flow Parameters (`--self_flow_args`)](#self-flow-parameters---self_flow_args)
      - [Notes](#notes)
    - [HFATO (High-Frequency Awareness Training Objective)](#hfato-high-frequency-awareness-training-objective)
      - [CLI Flags](#cli-flags-1)
      - [HFATO Parameters (`--hfato_args`)](#hfato-parameters---hfato_args)
      - [Relay LoRA Workflow (Image-Only Training)](#relay-lora-workflow-image-only-training)
    - [Latent Temporal Objectives](#latent-temporal-objectives)
      - [Weighting Args](#weighting-args)
      - [Delta Loss Args](#delta-loss-args)
  - [Conditioning & Control](#conditioning--control)
    - [Composable Conditioning Recipe](#composable-conditioning-recipe)
    - [Intrinsic Conditions](#intrinsic-conditions)
      - [Spatial-Crop Region Conditioning (outpaint)](#spatial-crop-region-conditioning-outpaint)
      - [On-Disk Mask Conditioning (inpaint/outpaint)](#on-disk-mask-conditioning-inpaintoutpaint)
      - [Video Extension (prefix/suffix)](#video-extension-prefixsuffix)
      - [Audio Extension (prefix/suffix)](#audio-extension-prefixsuffix)
      - [Audio Inpaint / Mask Conditioning](#audio-inpaint--mask-conditioning)
    - [Reference Conditioning (IC-LoRA)](#reference-conditioning-ic-lora)
      - [IC-LoRA / Video-to-Video Training](#ic-lora--video-to-video-training)
        - [Step 1: Prepare Dataset](#step-1-prepare-dataset)
        - [Step 2: Dataset Config](#step-2-dataset-config)
        - [Step 3: Cache Latents](#step-3-cache-latents)
        - [Step 4: Cache Text Encoder Outputs](#step-4-cache-text-encoder-outputs)
        - [Step 5: Train](#step-5-train)
        - [Step 6: Sample Prompts](#step-6-sample-prompts)
        - [IC-LoRA Arguments](#ic-lora-arguments)
        - [IC-LoRA Dataset Config Options](#ic-lora-dataset-config-options)
        - [Notes](#notes-1)
        - [Recommended settings](#recommended-settings)
        - [How it works](#how-it-works-2)
      - [Audio-Reference IC-LoRA](#audio-reference-ic-lora)
        - [Step 1: Prepare Data](#step-1-prepare-data)
        - [Step 2: Dataset Config](#step-2-dataset-config-1)
        - [Step 3: Cache Latents](#step-3-cache-latents-1)
        - [Step 4: Cache Text Encoder Outputs](#step-4-cache-text-encoder-outputs-1)
        - [Step 5: Train](#step-5-train-1)
        - [Step 6: Sample Prompts](#step-6-sample-prompts-1)
        - [Audio-Reference IC-LoRA Arguments](#audio-reference-ic-lora-arguments)
        - [Audio-Reference Dataset Config Options](#audio-reference-dataset-config-options)
        - [Notes](#notes-2)
    - [Directional Training (A2V / V2A)](#directional-training-a2v--v2a)
    - [Latent Guides](#latent-guides)
      - [Latent Guide Dataset Config Options](#latent-guide-dataset-config-options)
      - [IC-LoRA Compatibility Matrix](#ic-lora-compatibility-matrix)
      - [Reference target-frame routing](#reference-target-frame-routing)
      - [Endpoint Keyframe Training](#endpoint-keyframe-training)
      - [Video Anchor Training](#video-anchor-training)
      - [Sample Prompt Flags](#sample-prompt-flags)
    - [Conditioning representation](#conditioning-representation)
  - [Slider LoRA Training](#slider-lora-training)
    - [Text-Only Mode](#text-only-mode)
      - [Slider Config (`ltx2_slider.toml`)](#slider-config-ltx2_slidertoml)
        - [Anchors (concept preservation)](#anchors-concept-preservation)
      - [Example Command (Text-Only)](#example-command-text-only)
      - [Text-Only Arguments](#text-only-arguments)
    - [Reference Mode](#reference-mode)
      - [Step 1: Prepare Paired Data](#step-1-prepare-paired-data)
      - [Step 2: Cache Latents and Text](#step-2-cache-latents-and-text)
      - [Step 3: Configure and Train](#step-3-configure-and-train)
        - [Slider Config (`ltx2_slider_reference.toml`)](#slider-config-ltx2_slider_referencetoml)
        - [Example Command (Reference)](#example-command-reference)
        - [Audio Reference Sliders](#audio-reference-sliders)
    - [IC-slider](#ic-slider)
      - [Slider Config (`ltx2_slider_ic_reference.toml`)](#slider-config-ltx2_slider_ic_referencetoml)
        - [Example Command (IC-Aware Reference)](#example-command-ic-aware-reference)
    - [Slider Tips](#slider-tips)
  - [Reinforcement-Learning Post-Training (RL-LoRA)](#reinforcement-learning-post-training-rl-lora)
    - [Overview](#overview)
    - [How to use](#how-to-use)
    - [Update rules](#update-rules)
    - [The reward zoo](#the-reward-zoo)
    - [Tips](#tips-1)
    - [Writing a custom reward](#writing-a-custom-reward)
  - [Appendix: Full-Parameter Fine-Tuning](#appendix-full-parameter-fine-tuning)
    - [Dataset Preparation](#dataset-preparation)
    - [Dense bf16 (Adafactor)](#dense-bf16-adafactor)
      - [Using Full-Finetune Checkpoints for Inference](#using-full-finetune-checkpoints-for-inference)
    - [BAdam Block-Coordinate](#badam-block-coordinate)
    - [Q-GaLore Quantized](#q-galore-quantized)
    - [APOLLO and QAPOLLO](#apollo-and-qapollo)
    - [Int8 / Int8 ConvRot / EMA](#int8--int8-convrot--ema)
    - [Optimizing VRAM Usage](#optimizing-vram-usage)
      - [Block Swap for Full Fine-Tuning](#block-swap-for-full-fine-tuning)
  - [References](#references)

---

# Part I — Getting Started

## Installation
<sub>[↑ contents](#table-of-contents)</sub>

The base installation procedure is the same as musubi-tuner — follow the [Installation guide](../README.md#installation) (`pip install -e .` in a virtual environment). The sections below cover LTX-2-specific requirements (CUDA version, model downloads) that go on top of the base install.

Windows users can also use [`scripts/install.ps1`](#windows-setup--update-script) as a setup/update helper; that section documents the script's current operations and options.

Command examples assume the repository root and an activated project environment. Ellipses and parenthesized phrases are placeholders for the remaining required arguments.

For a Windows-focused community setup example for this fork (tested environment and install helpers), see [Discussion #19: Windows OS installation/usage helpers](https://github.com/AkaneTendo25/musubi-tuner/discussions/19).

### CUDA Version
<sub>[↑ contents](#table-of-contents)</sub>

The PyTorch install command must use a CUDA version compatible with your GPU. Adjust the `--index-url` accordingly:

```bash
# Most GPUs — RTX 30xx / 40xx / 50xx (matches the installer default, cu128):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

RTX 50xx (Blackwell) requires cu128 on Windows; cu126 is not supported on those GPUs. Always match the CUDA version to your GPU architecture — check [PyTorch's compatibility matrix](https://pytorch.org/get-started/locally/) for the latest supported versions.

After installing the project, verify the PyTorch package trio, compiled operators, audio transforms, and CUDA runtime:

```bash
python -m musubi_tuner.verify_environment --require-cuda --expected-cuda cu128
```

Replace `cu128` with the CUDA extra you installed. Add `--json` for machine-readable output. The command exits unsuccessfully when the Torch, TorchVision, and TorchAudio release lines or CUDA builds do not match.

### Downloading Required Models
<sub>[↑ contents](#table-of-contents)</sub>

> [!WARNING]
> The dashboard UI and the Windows Setup / Update script are under active testing. Their behavior may change, and some local environments may still require manual fixes.

The dashboard can handle common model downloads directly:

- Use the project page to choose a template that matches your use case.
- Open `Caching`, `Training`, or `Inference` and use the download actions beside the LTX-2 and Gemma model fields.
- Use `Setup & Updates` in the dashboard to verify the install, repo state, shortcuts, and project readiness before you start caching or training.

Manual download is still supported, and is useful for managing checkpoints outside the default model directory.

**LTX-2 Checkpoint** — use as `--ltx2_checkpoint`:
- LTX-2 (19B): [ltx-2-19b-dev.safetensors](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors)
- LTX-2.3 (22B): [ltx-2.3-22b-dev.safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)

**Gemma Text Encoder** — pick one:
- HF directory (`--gemma_root`): [gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/Lightricks/gemma-3-12b-it-qat-q4_0-unquantized)
- Single file (`--gemma_safetensors`): [gemma_3_12B_it_fp8_e4m3fn.safetensors](https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors)

Other Gemma 3 12B variants may work but not all have been tested.

---

## Quick Start
<sub>[↑ contents](#table-of-contents)</sub>

A minimal end-to-end run. Each step links to its full reference.

1. **Install** the trainer and download the model — see [Installation](#installation).
2. **Prepare a dataset**: a folder of clips (or images) with a `.txt` caption next to each file, plus a `dataset.toml` — see [Dataset Configuration](#dataset-configuration).
3. **Cache latents**: run `ltx2_cache_latents.py` over the dataset — see [Caching Latents](#caching-latents).
4. **Cache text encoder outputs**: run `ltx2_cache_text_encoder_outputs.py` — see [Caching Text Encoder Outputs](#caching-text-encoder-outputs).
5. **Train**: launch `ltx2_train_network.py` with the standard LoRA command — see [Training Your First LoRA](#training-your-first-lora).
6. **Sample** during training, or run your finished LoRA standalone — see [Sampling, Checkpoints & Resuming](#sampling-checkpoints--resuming) and [Standalone Inference](#standalone-inference).

Everything beyond a basic LoRA run — quantization, optimizers, regularizers, conditioning (IC-LoRA, latent guides), sliders, RL, and full-parameter fine-tuning — is in [Part III — Advanced](#part-iii--advanced).

## Supported Dataset Types
<sub>[↑ contents](#table-of-contents)</sub>

| Mode | Dataset Type | Notes |
|------|--------------|-------|
| `video` | Images | Treated as 1-frame samples (`F=1`) |
| `video` | Videos | Standard video training |
| `av` | Videos with audio | Audio extracted from video or external audio files |
| `audio` | Audio only | Dataset must be audio-only; training uses audio-driven latent geometry |

---

## Dataset Configuration
<sub>[↑ contents](#table-of-contents)</sub>

The dataset config is a TOML file with `[general]` defaults and `[[datasets]]` entries. Common options shared across all musubi-tuner architectures — including `frame_extraction` modes, JSONL metadata format, control image support, and resolution bucketing — are documented in the [Dataset Configuration guide](./dataset_config.md). The options below are LTX-2-specific or supplement base defaults.

**Captions.** Each clip needs a caption: a `.txt` file with the same stem beside the media (set with `caption_extension`), or a `caption` field in a JSONL. Write a detailed prose description of the visual content; for `av` datasets, also describe the audio (speech, music, ambient). Review machine-generated captions before training. Authoring captions is outside this trainer — see the dataset and captioning tools in [References](#references).

### Image Dataset Notes
<sub>[↑ contents](#table-of-contents)</sub>

Image datasets use the common image schema from [Dataset Configuration](./dataset_config.md), including
`image_directory` or `image_jsonl_file`. For LTX-2 IC-LoRA with image datasets, use `reference_directory`
and `reference_cache_directory`. Internally this is normalized onto the shared image control path, so
other non-IC image workflows can continue using `control_directory`.

### Video Dataset Options
<sub>[↑ contents](#table-of-contents)</sub>

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `video_directory` | string | — | Path to video directory |
| `video_jsonl_file` | string | — | Path to JSONL metadata file |
| `resolution` | int or [int, int] | [960, 544] | Target resolution |
| `target_frames` | [int] | [1] | List of target frame counts. Rounded down to the nearest valid value at cache time (`8·k + 1`: 1, 9, 17, 25, 33, 41, 49, …), since the VAE encodes the first frame alone then in groups of 8 — e.g. `50` caches as `49`. |
| `frame_extraction` | string | `"head"` | Frame extraction mode |
| `max_frames` | int | 129 | Maximum number of frames |
| `source_fps` | float | auto-detected | Source video FPS. Auto-detected from video container metadata when not set. Use this to override auto-detection. |
| `target_fps` | float | 25.0 | Target training FPS. Frames are resampled to this rate. When audio is present and its duration differs from the target video duration by more than ~1%, the waveform is time-stretched (pitch-preserving) to match; disable with `--preserve_audio_timing`. |
| `batch_size` | int | 1 | Batch size |
| `bucket_batch_sizes` | table | — | Opt-in exact batch-size overrides keyed by `WIDTHxHEIGHTxFRAMES`; unspecified buckets continue using `batch_size`. Larger values can improve short-bucket throughput but require more VRAM. |
| `num_repeats` | int | 1 | Dataset repetitions |
| `enable_bucket` | bool | false | Enable resolution bucketing |
| `bucket_no_upscale` | bool | false | Prevent upscaling when bucketing (only downscale to fit) |
| `cache_directory` | string | — | Latent cache output directory |
| `reference_directory` | string | — | Reference images/videos for IC-LoRA (matched by filename) |
| `reference_cache_directory` | string | — | Output directory for cached reference latents (IC-LoRA) |
| `reference_audio_directory` | string | — | Reference audio files for audio-reference IC-LoRA (matched by filename stem) |
| `reference_audio_cache_directory` | string | — | Output directory for cached reference audio latents |
| `separate_audio_buckets` | bool | false | Keep audio/non-audio items in separate batches |
| `loss_mask_directory` | string | — | Optional stem-matched image/video masks for masked loss |
| `default_loss_mask_path` | string | — | Optional fallback image/video mask |
| `loss_mask_use_alpha` | bool | false | Image datasets only: use target image alpha as the mask when no mask directory is set |
| `loss_mask_invert` | bool | false | Invert image/video masks before caching |


### Audio Dataset Options
<sub>[↑ contents](#table-of-contents)</sub>

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
| `loss_mask_directory` | string | — | Optional stem-matched JSON/TXT/CSV interval masks |
| `default_loss_mask_path` | string | — | Optional fallback interval mask file |

### Masked Loss Datasets
<sub>[↑ contents](#table-of-contents)</sub>

Masked loss is configured in the dataset TOML and applied automatically during latent/audio caching. Training then reads `video_loss_mask` and `audio_loss_mask` from the cache and multiplies them with any masks already required by the selected training mode. This means masked loss works with standard video/image training, first-frame conditioning, v2v / IC-LoRA variants, audio-only training, and AV training. In IC-LoRA modes, masks are applied to target loss only; reference and conditioning tokens remain excluded from loss where the IC mode already excludes them.

If no mask options are set, the cache output and training behavior are unchanged.

Image/video masks:
- White means full loss, black means no loss, and grayscale values are soft weights.
- `loss_mask_directory` matches masks by source filename stem.
- JSONL datasets may use per-item `loss_mask_path` / `image_loss_mask_path` / `video_loss_mask_path`.
- A single image mask can be used for all frames, or a video/image-sequence mask can be used for frame-varying masks.

Audio masks:
- Use intervals in seconds via JSONL `loss_mask_intervals` / `audio_loss_mask_intervals`.
- Or set `loss_mask_path` / `audio_loss_mask_path` to a JSON/TXT/CSV interval file.
- Directory datasets can use `loss_mask_directory` with stem-matched interval files.

Examples:

```toml
[[datasets]]
video_directory = "videos"
cache_directory = "cache"
caption_extension = ".txt"
target_frames = [33]
loss_mask_directory = "video_masks"
```

```json
{"audio_path":"audio/line.wav","caption":"voice","loss_mask_intervals":[[0.25, 1.8], [2.4, 3.1]]}
```

When a loss mask is configured, the training loop logs the mask-active fraction in the progress bar (`mv_act` / `ma_act`) and emits `loss/video_mask_active`, `loss/video_loss_unmasked`, `loss/video_loss_masked` (and audio equivalents) to TensorBoard / WandB. The pre- vs post-mask loss values show whether the mask is shifting the gradient.

The same video mask metrics can also appear without dataset mask files when an intrinsic conditioner excludes clean conditioning tokens from denoising loss. The common default case is first-frame conditioning: `--ltx2_first_frame_conditioning_p` defaults to `0.1`, and when it fires on a multi-frame video, latent frame 0 is kept clean and excluded from loss. This is expected and does not mean dataset masked-loss training is enabled. Set `--ltx2_first_frame_conditioning_p 0.0` only if you want pure T2V with no first-frame conditioning.

### Example TOML
<sub>[↑ contents](#table-of-contents)</sub>

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
bucket_batch_sizes = { "512x512x33" = 2 }
target_fps = 25    # optional, defaults to 25
```

`bucket_batch_sizes` never selects values automatically. Keep
`gradient_accumulation_steps × selected bucket batch size` consistent with the
intended effective batch when comparing or combining bucket configurations.

### Frame Rate (FPS) Handling
<sub>[↑ contents](#table-of-contents)</sub>

During latent caching, the source FPS is **auto-detected** from each video's container metadata and frames are resampled to `target_fps` (default: 25). The model receives video at the configured temporal rate regardless of the source material.

#### How It Works
<sub>[↑ contents](#table-of-contents)</sub>

1. For each video file, the source FPS is read from the container metadata (`average_rate` or `base_rate`).
2. If `abs(ceil(source_fps) - target_fps) > 1`, frames are resampled (dropped) to match `target_fps`.
3. If the difference is within this threshold (e.g., 23.976 → ceil=24 vs target 25, diff=1), no resampling is done — this avoids spurious frame drops from NTSC rounding (23.976, 29.97, 59.94, etc.).
4. If audio is present (`--ltx2_mode av`), the audio waveform is time-stretched (pitch-preserving) to match the resampled video duration — but only when the duration mismatch exceeds ~1%, and skipped if `--preserve_audio_timing` is set.

#### Common Scenarios
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

During latent caching, log messages report the resampling decision for each video:
```
Auto-detected source FPS: 60.00 for my_video.mp4
Resampling my_video.mp4: 60.00 FPS -> 25.00 FPS
```
The `Resampling` line is shown at the default log level; the `Auto-detected source FPS` line is debug-level and appears only with increased log verbosity. If you see **no** "Resampling" line for a video, it means source and target FPS were close enough (within 1 FPS after rounding up the source) and all frames were kept as-is. If you see unexpected frame counts in your cached latents, check the `Resampling` log lines first.

#### Quick Reference
<sub>[↑ contents](#table-of-contents)</sub>

| Your situation | What to set | What happens |
|---|---|---|
| Mixed FPS dataset, want 25fps training | Nothing (defaults work) | Each video auto-detected, resampled to 25fps |
| All videos are 25fps | Nothing | Auto-detected as 25fps, no resampling |
| All videos are 60fps, want 60fps training | `target_fps = 60` | Auto-detected as 60fps, no resampling |
| All videos are 60fps, want 25fps training | Nothing | Auto-detected as 60fps, resampled to 25fps |
| VFR videos with wrong detection | `source_fps = 30` (your actual FPS) | Overrides auto-detection |
| Image directory | Nothing | No FPS concept, all images loaded |

---

## Caching Latents
<sub>[↑ contents](#table-of-contents)</sub>

This step pre-processes media files into VAE latents to speed up training.

**Script:** `ltx2_cache_latents.py`

### Latent Caching Command
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

- `--ltx2_mode`, `--ltx_mode`: Caching modality selector. Default is video-only (`v`/`video`). Use `av` to cache both `*_ltx2.safetensors` (video) and `*_ltx2_audio.safetensors` (audio) latents.
- `--ltx2_audio_source video|audio_files`: Use audio from the video or from external files.
- `--ltx2_audio_dir`, `--ltx2_audio_ext`: Optional when using `--ltx2_audio_source audio_files` (default extension: `.wav`).
- `--ltx2_checkpoint`: Required for `--ltx2_mode av` or `--ltx2_mode audio`.
- `--audio_only_sequence_resolution`: Virtual square pixel resolution used to derive the audio-only sequence length for `shifted_logit_normal` timestep sampling in `--ltx2_mode audio` (default: `64`; set `0` to use the cached virtual geometry instead). See [Timestep Sampling](#timestep-sampling).
- `--audio_only_target_resolution`: Optional square override for audio-only latent geometry. Only takes effect when `--audio_only_sequence_resolution 0`; otherwise the fixed sequence resolution is used instead.
- `--audio_only_target_fps`: Target FPS used to derive audio-only frame counts from audio duration (default: `25`).
- `--audio_video_latent_channels`: Optional override for audio-only video latent channels (auto-detected from checkpoint by default).
- `--ltx2_audio_dtype`: Data type for audio VAE encoding (default: `float16`).
- `--audio_video_latent_dtype`: Optional override for audio-only video latent dtype (defaults to `--ltx2_audio_dtype`).
- `--vae_dtype`: Data type for VAE latents (default comes from the cache script).
- `--video_decode_backend pyav|decord|torchcodec`: Opt-in decode backend. Omitting the flag keeps the `pyav` path. The `decord` and `torchcodec` adapters batch-decode the selected source-frame indices and require their optional runtime dependencies. If an alternate-backend decode raises, the loader logs a warning and retries that item with PyAV; a subsequent PyAV error still fails normally. Different decoder/codec implementations can produce different decoded pixels and therefore different caches, so keep one backend fixed when cache reproducibility matters. The CLI overrides `LTX2_VIDEO_DECODE_BACKEND` when supplied. Decord receives `LTX2_DECORD_NUM_THREADS` (default `1`) for each reader.
- `--video_decode_device cpu|cuda`: Device passed to TorchCodec's `VideoDecoder` (default `cpu`); ignored by other backends. CUDA support depends on the installed TorchCodec wheel, FFmpeg libraries, and hardware. The CLI overrides `LTX2_VIDEO_DECODE_DEVICE` when supplied.
- `--cpu_staged_checkpoint_loading`: Opt-in compatibility mode. Opens LTX checkpoint tensors on CPU and then moves them synchronously to the selected device instead of loading them directly on that device. This may work around direct CUDA safetensors loading errors in some environments. Initial model loading is slower and temporarily uses additional CPU RAM; caching speed after loading is unaffected.
- `--save_dataset_manifest`: Optional. Saves a cache-only dataset manifest for source-free training.
- `--precache_sample_latents`: Cache I2V / V2V / reference-audio conditioning latents for sample prompts, then continue with normal latent caching. Requires `--sample_prompts`.
- `--sample_latents_cache`: Path for the I2V conditioning latents cache file (default: `<cache_dir>/ltx2_sample_latents_cache.pt`).
- `--reference_frames`: Number of reference frames to cache for IC-LoRA / V2V (default: `1`). With `--skip_existing`, an existing reference cache is re-encoded automatically when its stored frame count no longer matches the requested `--reference_frames` / `--reference_temporal_scale`.
- `--reference_downscale`: Spatial downscale factor for cached reference latents (default: `1`).
- `--reference_temporal_scale`: Temporal subsample factor for cached references (default: `1`). `2` keeps the first frame then every other frame, storing the reference at a lower temporal resolution; its time positions are rescaled to the target at train/inference. Requires `--reference_frames` = `8*k + 1` with the factor dividing `k`, and the reference must span the same number of frames as the target. Validated at cache time.
- `--atomic_cache_writes`: Opt-in safety mode. Writes cache files to a temporary sibling file first, then replaces the final cache path only after a successful save.

### Latent Cache Output Files
<sub>[↑ contents](#table-of-contents)</sub>

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2.safetensors` | Video latents: `latents_{F}x{H}x{W}_{dtype}`. If masked loss is configured, also stores `video_loss_mask`. In audio-only mode, this file also stores `ltx2_virtual_num_frames_int32`, `ltx2_virtual_height_int32`, and `ltx2_virtual_width_int32` — the virtual spatial geometry used for timestep sampling, derived from `--audio_only_sequence_resolution` when `> 0`, or from the dataset/target resolution when `0`. |
| `*_ltx2_audio.safetensors` | Audio latents: `audio_latents_{T}x{mel_bins}x{channels}_{dtype}`, `audio_lengths_int32`. If audio masked loss is configured, also stores `audio_loss_mask`. |

### Latent Cache Preview and Verification
<sub>[↑ contents](#table-of-contents)</sub>

`src/musubi_tuner/ltx2_cache_preview.py` checks cached LTX-2 latent files and writes `summary.json` with tensor keys, shapes, dtypes, metadata, and NaN/Inf results. Add `--checkpoint` only when decoded MP4/PNG/WAV previews are needed.

```bash
python src/musubi_tuner/ltx2_cache_preview.py ^
  --input /path/to/cache_dir ^
  --output /path/to/cache_preview ^
  --stats
```

Useful flags: `--checkpoint`, `--limit N`, `--fail_on_error`, `--no_decode`, `--fps 25`.
For an AV cache that must be complete, add
`--require_companions video,audio,text`; the full directory is inventoried
even when `--limit` selects only a deterministic preview subset.
Newly written LTX-2 video and audio latent caches also record the source path,
file size, and nanosecond modification time. Add `--check_source` to report a
changed or missing source as an error. Older caches without these source details
remain readable and receive a warning because their freshness cannot be
verified.
The verifier also checks that latent tensor shapes match the geometry declared
in their keys. Paired video/audio caches are checked for architecture and
duration consistency; `--av_duration_tolerance` controls the allowed duration
difference and defaults to 0.05 seconds. Newly generated caches contain the
required FPS and duration metadata, while older pairs receive a warning when
duration cannot be established.
The raw safetensors metadata remains available, and every `summary.json` entry
also exposes normalized `cache_key`, `architecture`, `source`, and `media`
objects. Video media reports latent geometry, frame count, target FPS, and
duration. Audio media reports latent/effective steps, mel bins, channels,
target FPS, and duration.

### Memory Optimization for Caching
<sub>[↑ contents](#table-of-contents)</sub>

On Out-Of-Memory (OOM) errors during caching (especially at higher resolutions like 1080p), use one of two options:

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

## Caching Text Encoder Outputs
<sub>[↑ contents](#table-of-contents)</sub>

This step pre-computes text embeddings using the Gemma text encoder.

**Script:** `ltx2_cache_text_encoder_outputs.py`

### Text Encoder Caching Command
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

- `--gemma_root`: Path to the local Gemma model folder (HuggingFace format). Required unless `--gemma_safetensors` is used.
- `--gemma_safetensors`: Path to a single Gemma `.safetensors` file (e.g. an FP8 export from ComfyUI). Loads weights, config, and tokenizer from one file — no `--gemma_root` needed. See [Loading Gemma from a Single Safetensors File](#loading-gemma-from-a-single-safetensors-file) below.
- `--gemma_load_in_8bit`: Loads Gemma in 8-bit quantization. Cannot be combined with `--gemma_safetensors`.
- `--gemma_load_in_4bit`: Loads Gemma in 4-bit quantization. Cannot be combined with `--gemma_safetensors`.
- `--gemma_bnb_4bit_quant_type nf4|fp4`: Quantization type for 4-bit loading (default: `nf4`).
- `--gemma_bnb_4bit_disable_double_quant`: Disable bitsandbytes double quantization for 4-bit loading.
- `--gemma_bnb_4bit_compute_dtype auto|fp16|bf16|fp32`: Compute dtype for 4-bit operations (default: `auto`, uses `--mixed_precision` dtype).
- `--ltx2_checkpoint`: Required. Use `--ltx2_text_encoder_checkpoint` to override for text encoder connector weights.
- `--cpu_staged_checkpoint_loading`: Opt-in compatibility mode for loading the LTX text-connector checkpoint through CPU before moving it to the selected device. This may work around direct CUDA safetensors loading errors in some environments. It does not change how separate `--gemma_root` or `--gemma_safetensors` weights are loaded.
- `--cache_before_connector`: Also save pre-connector text features (`video_features_{dtype}`, `audio_features_{dtype}`) alongside standard post-connector embeddings. Required for `--train_connectors` during training. Does not change standard cache keys; only adds extra tensors.
- `--atomic_cache_writes`: Opt-in safety mode. Writes text and prompt cache files to a temporary sibling file first, then replaces the final cache path only after a successful save.
- 8-bit/4-bit loading requires `--device cuda`.

> [!IMPORTANT]
> `--ltx2_mode` / `--ltx_mode` **must match** the mode used during latent caching. Default is `video`; use `av` to concatenate video and audio prompt embeddings.

### Text Encoder Output Files
<sub>[↑ contents](#table-of-contents)</sub>

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2_te.safetensors` | `video_prompt_embeds_{dtype}`, `audio_prompt_embeds_{dtype}` (av only), `prompt_attention_mask`, `text_{dtype}`, `text_mask` |
| (with `--cache_before_connector`) | Above keys plus `video_features_{dtype}`, `audio_features_{dtype}` (av only) |

### Loading Gemma from a Single Safetensors File
<sub>[↑ contents](#table-of-contents)</sub>

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

## Training Your First LoRA
<sub>[↑ contents](#table-of-contents)</sub>

Launch the training loop using `accelerate`.

**Script:** `ltx2_train_network.py`

### Choosing Model Version for Training (2.0 vs 2.3)
<sub>[↑ contents](#table-of-contents)</sub>

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
- E4M3 LoRA fusion uses optional Triton stochastic rounding only on compute capability 8.9+. Unsupported GPUs, including Ampere, use the existing bf16 merge fallback; Triton is not required.

Example (LTX-2.3 training):
```bash
--ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
--ltx_version 2.3 ^
--ltx_version_check_mode error
```

### Standard LoRA Training
<sub>[↑ contents](#table-of-contents)</sub>

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
  --max_train_steps 2000 ^
  --save_every_n_epochs 5 ^
  --save_last_n_epochs 5 ^
  --save_state ^
  --save_state_on_train_end ^
  --output_dir output ^
  --output_name ltx23_lora
```

`--max_train_steps` sets the run length (otherwise it defaults to 1600). `--save_every_n_epochs` writes intermediate checkpoints; `--save_state` (with `--save_state_on_train_end`) additionally writes resumable training state so a run can continue with `--resume` — see [Resuming Training](#resuming-training). To run the finished LoRA on its own afterwards, see [Standalone Inference](#standalone-inference).

Pre-quantized FP8 checkpoints (`*fp8*.safetensors`) are supported — `--fp8_base --fp8_scaled` works the same as with standard checkpoints (weights are dequantized to bf16 first, then re-quantized).

Optional torch.compile and DataLoader flags:

| Flag | Effect |
|---|---|
| `--compile` | Enables block-level `torch.compile` for the LTX-2 transformer. |
| `--compile_mode MODE` | `torch.compile` mode, e.g. `default` or `max-autotune-no-cudagraphs`. |
| `--compile_cache_size_limit N` | Sets `torch._dynamo.config.cache_size_limit`, the maximum number of cached variants before Dynamo's cache-limit behavior applies. It does not prevent recompilation for new shapes. |
| `--inductor_config KEY=VALUE ...` | With `--compile`, sets existing `torch._inductor.config` / `torch._dynamo.config` attributes; dotted keys are accepted. Values parse as bool/int/float/none/string and unknown keys are skipped with a warning. The available keys and their effect belong to the installed PyTorch build; for example, `triton.enable_persistent_tma_matmul` is forwarded only when that attribute exists. |
| `--compile_dynamic false` | Avoids dynamic-shape compile paths when the training shape is fixed. |
| `--compile_auto_cache_size_limit` | Raises the Dynamo cache limit based on the number of compiled blocks. |
| `--compile_fallback_to_eager` | Restores eager blocks and continues if compile setup fails. |
| `--compile_cudagraph_mark_step` | Calls the PyTorch CUDAGraph step marker each train step; use only after measuring. |
| `--dynamo_use_regional_compilation` | Requests Accelerate regional compilation when supported by the installed Accelerate version. |
| `--dataloader_pin_memory` | Enables pinned host buffers and non-blocking host-to-device copies. |
| `--dataloader_prefetch_factor N` | Sets DataLoader worker prefetching when workers are enabled. |

All flags above are off by default. `torch.compile` speedups are workload-dependent; gradient checkpointing and very small step counts can hide gains behind compile overhead. For generic compile arguments and limitations, see [torch.compile](./torch_compile.md).

For LTX-2 checkpoints, replace:
- `--ltx2_checkpoint /path/to/ltx-2.3.safetensors` -> `--ltx2_checkpoint /path/to/ltx-2.safetensors`
- `--ltx_version 2.3` -> `--ltx_version 2.0`

> For DoRA, LyCORIS, quantization, optimizers, per-module settings, regularizers, conditioning (IC-LoRA / latent guides), and every other option, see [Part III — Advanced](#part-iii--advanced).

### Audio-Only Training
<sub>[↑ contents](#table-of-contents)</sub>

Audio-only training fits a LoRA on an audio-only dataset — the model generates audio with no video target. The flow mirrors standard LoRA training with `--ltx2_mode audio` end to end:

1. **Dataset**: build an audio-only dataset — see [Audio Dataset Options](#audio-dataset-options).
2. **Cache latents**: run `ltx2_cache_latents.py` with `--ltx2_mode audio` (and `--ltx2_checkpoint`, which is required in audio mode). The audio-only sequence length used for timestep sampling comes from `--audio_only_sequence_resolution` (default `64`) — see [Caching Latents](#caching-latents).
3. **Cache text encoder outputs**: same as any run — see [Caching Text Encoder Outputs](#caching-text-encoder-outputs).
4. **Train**: use the [Standard LoRA](#standard-lora-training) command with `--ltx2_mode audio` instead of `av`. Select the audio modules with `--lora_target_preset` — see [LoRA Targets](#lora-targets) — and tune audio supervision with the [Additional Audio Training Flags](#additional-audio-training-flags).

Sampling produces audio-only previews (`--sample_audio_only`). Audio-only timestep sampling uses `shifted_logit_normal` with the virtual geometry described under [Timestep Sampling](#timestep-sampling).

# Part II — Core Workflows

## Sampling, Checkpoints & Resuming
<sub>[↑ contents](#table-of-contents)</sub>

This section covers sampling **during** training plus checkpoint and resume handling. To run a finished LoRA on its own, see [Standalone Inference](#standalone-inference). Conditioning-specific sample inputs (reference / latent-guide / anchor paths in prompt files) are documented with their features in [Part III — Advanced](#part-iii--advanced).

### Sampling with Tiled VAE
<sub>[↑ contents](#table-of-contents)</sub>

The prompt file format (`--sample_prompts`) — including guidance scale, negative prompt, and per-prompt inference parameters — is documented in the [Sampling During Training guide](./sampling_during_training.md). LTX-2 extends this with `--v <path>` (IC-LoRA reference), `--ra <path>` (audio-reference IC-LoRA), and `--im <path>` / `--ii <0|1>` / `--it <float>` (inpaint mask / invert / threshold, composed with a `--v` reference — see [Step 6: Sample Prompts](#step-6-sample-prompts)) prompt-line options. Put these options after the prompt text.

| Argument | Default | Description |
|----------|---------|-------------|
| `--height` | 512 | Base sample output height. With the default sampling preset, LTX-2.3 uses `512` unless the prompt line sets `--h` |
| `--width` | 768 | Base sample output width. With the default sampling preset, LTX-2.3 uses `768` unless the prompt line sets `--w` |
| `--sample_num_frames` | 45 | Base sample frame count. With the default sampling preset, LTX-2.3 uses `121` unless the prompt line sets `--f` |
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
| `--sample_sampling_preset` | `defaults` | Validation sampling preset. Named values: `defaults` (resolves per `--ltx_version`), `legacy` (bypass preset defaults), `ltx20`, `ltx23`, `ltx23_hq` (full-quality LTX-2.3), and `distilled_two_stage` (distilled two-stage sampling, see [Two-Stage Sampling](#two-stage-sampling)). For `--ltx_version 2.3`, `defaults` resolves to the LTX-2.3 defaults (`30` steps, `768x512`, `121` frames, CFG/STG defaults, CFG rescale `0.7`) |
| `--sample_sampler` | `auto` | Denoising sampler. `auto` uses `res_2s` for full LTX presets and Euler for `distilled_two_stage` |
| `--sample_sigma_schedule` | `auto` | Sigma schedule. `auto` uses latent-aware LTX shifted sigmas and the exact LTX-2.3 distilled schedule for the distilled preset |

### Precached Sample Prompts
<sub>[↑ contents](#table-of-contents)</sub>

To avoid loading Gemma during training for sample generation, precache the prompt embeddings:

1. During text encoder caching, add `--precache_sample_prompts --sample_prompts sampling_prompts.txt` to also cache the sample prompt embeddings.
2. During training, add `--use_precached_sample_prompts` (or `--precache_sample_prompts`) to load embeddings from cache instead of running Gemma.
- `--sample_prompts_cache`: Path to the precached embeddings file. Defaults to `<cache_directory>/ltx2_sample_prompts_cache.pt`.

For IC-LoRA / V2V training, also precache the conditioning image latents during latent caching (see [Latent Caching Arguments](#latent-caching-arguments)):
1. During latent caching, add `--precache_sample_latents --sample_prompts sampling_prompts.txt`.
2. During training, add `--use_precached_sample_latents` to load conditioning latents from cache instead of loading the VAE encoder.
- `--sample_latents_cache`: Path to the precached latents file. Defaults to `<cache_directory>/ltx2_sample_latents_cache.pt`.
- Rebuild this cache after changing sample prompt `--w`, `--h`, or `--reference_downscale`; cached video conditioning latent shapes are validated and stale spatial shapes are rejected. Rebuild manually after changing `--i` or `--v`, because cache entries are matched by prompt index rather than by source path.

### Two-Stage Sampling
<sub>[↑ contents](#table-of-contents)</sub>

> [!NOTE]
> This feature is disabled by default. Two-stage inference reduces the spatial axis when a spatial upsampler is supplied and reduces the temporal axis when a temporal upsampler is supplied, then upsamples and refines. For example, a spatial stage at final size `512x512` uses `256x256` in stage 1. Compare outputs with the single-stage path for the intended checkpoint and settings.

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample_two_stage` | off | Enable two-stage inference during sampling |
| `--spatial_upsampler_path` | — | Path to a spatial upsampler model. Required unless `--temporal_upsampler_path` is set |
| `--temporal_upsampler_path` | — | Path to a temporal upsampler model. Can be used alone or with `--spatial_upsampler_path` |
| `--distilled_lora_path` | — | Path to distilled LoRA for stage refinement. External-format LTX-2 LoRAs are converted automatically |
| `--sample_stage2_steps` | 3 | Number of denoising steps for stage 2 |
| `--sample_stage1_distilled_lora_multiplier` | auto | Optional stage-1 distilled LoRA strength. With `res_2s`, auto uses `0.25`; with Euler, auto uses `0.0` |
| `--sample_stage2_distilled_lora_multiplier` | auto | Optional stage-2 distilled LoRA strength. With `res_2s`, auto uses `0.5`; with Euler, auto uses `1.0` |

Upsampler axes are read from checkpoint config. With `--temporal_upsampler_path`, CLI `--frame_count` and `--frame_rate` describe the final output; stage 1 runs at `(frame_count - 1) / 2 + 1` frames and half fps, then the temporal upsampler maps latent frames `F -> 2F - 1`. With the current LTX-2 video VAE, `(frame_count - 1)` must be divisible by `16`. If both spatial and temporal upsamplers are provided, temporal-only upsampling is applied before spatial upsampling, followed by one stage-2 denoising pass at the final geometry. Audio with temporal upsampling has not been validated.

LTX-2.3 preview starting point:

```bash
--sample_sampling_preset ltx23 ^
--sample_sampler auto ^
--sample_sigma_schedule auto
```

Prompt-level `--w`, `--h`, `--f`, and `--s` values override preset defaults. For LTX-2.3 presets, explicit `--w`/`--h` whose short side is below the preset's short side, or whose aspect ratio differs from the preset's aspect by more than `0.08`, and explicit `--s` values different from the preset step count, are logged as warnings.

### Checkpoint Output Format
<sub>[↑ contents](#table-of-contents)</sub>

Saved LoRA checkpoints are converted to ComfyUI format by default. Both the original musubi-tuner format and the ComfyUI format are kept. For standalone LTX-2 conversion, use `python -m musubi_tuner.ltx2_convert_lora_to_comfy`.

| Flag | Behavior |
|------|----------|
| *(default)* | Saves both `*.safetensors` (original) and `*.comfy.safetensors` (ComfyUI). |
| `--no_save_original_lora` | Deletes the original after conversion, keeping only `*.comfy.safetensors`. |
| `--no_convert_to_comfy` | Saves only the original `*.safetensors` (no conversion). |
| `--save_checkpoint_metadata` | Saves a `.json` sidecar file alongside each checkpoint with loss, lr, step, and epoch. |

> **Important:** Training can only be resumed from the **original** (non-comfy) checkpoint format. If you plan to use `--resume`, do not use `--no_save_original_lora`.
> ComfyUI-only LoRA files can still be used for warm-starting via `--network_weights`, `--base_weights`, or `--dim_from_weights`; only full `--resume` requires the original checkpoint plus saved training state.

For DoRA LoRA and DoKr LoKr, keep the original `*.safetensors` file for Musubi loading and resume. The training-time ComfyUI export is intended for ComfyUI and stores `dora_scale`; converting that file back to native Musubi DoRA/DoKr requires base-weight information that is not present in the standalone ComfyUI checkpoint.

DoRA-OFT and DoKr-OFT exports preserve the native OFT tensors (`.oft_R.*` plus OFT metadata). DoRA-OFT reverse conversion can keep the native `dora_scale`; DoKr-OFT reverse conversion requires the original base transformer to reconstruct Musubi's `lora_magnitude_vector.weight` exactly.

```bash
# Musubi -> ComfyUI, standalone
python -m musubi_tuner.ltx2_convert_lora_to_comfy output/adapter.safetensors ^
  --output output/adapter.comfy.safetensors ^
  --base_model /models/ltx-2.3-22b-dev.safetensors ^
  --dora_ff_only

# ComfyUI -> Musubi, standalone
python -m musubi_tuner.ltx2_convert_comfy_to_musubi ^
  --input output/adapter.comfy.safetensors ^
  --output output/adapter.musubi.safetensors ^
  --base_model /models/ltx-2.3-22b-dev.safetensors
```

The dashboard **Tools** tab includes a **ComfyUI Conversion** panel for one-shot conversion of an existing Musubi adapter. It uses the loaded project's LTX-2 mode, audio-only, FP8/W8A8, NF4, and quantization settings when a base transformer is needed for DoRA/DoKr conversion.

### Extract LoRA from Full Fine-Tune
<sub>[↑ contents](#table-of-contents)</sub>

`ltx2_extract_lora.py` extracts a native Musubi LoRA or DoRA adapter from a pair of full transformer checkpoints by factorizing the weight delta between the base model and the fine-tuned model. This is an offline utility; it does not change training or inference defaults.

```bash
python -m musubi_tuner.ltx2_extract_lora ^
  --base_model /models/ltx-2.3-22b-dev.safetensors ^
  --finetuned_model /models/my_full_finetune.safetensors ^
  --save_to output/extracted_lora.safetensors ^
  --target_preset full ^
  --rank_mode fro ^
  --fro_target 0.98 ^
  --max_rank 128 ^
  --save_precision bf16 ^
  --report_json output/extracted_lora.report.json
```

The default extraction path uses per-layer SVD with adaptive Frobenius-energy rank selection. Each selected 2D weight gets its own rank up to `--max_rank`, and the saved checkpoint uses normal native keys (`lora_down`, `lora_up`, `alpha`) so existing LTX-2 LoRA loading can infer per-module ranks from the file.

| Option | Default | Description |
|--------|---------|-------------|
| `--extract_mode lora|dora` | `lora` | Plain LoRA delta extraction, or DoRA direction/magnitude extraction |
| `--rank_mode fixed|fro|quantile|knee|relative_drop` | `fro` | Per-layer rank selection method |
| `--dim` | `64` | Rank for `fixed`, or low-rank SVD hint |
| `--min_rank` / `--max_rank` | `1` / `128` | Per-layer rank bounds |
| `--fro_target` | `0.98` | Squared singular-value energy target for `fro` mode |
| `--target_preset` | `full` | Same LTX-2 target preset names used by LoRA training, or `custom` with regex includes |
| `--connector_lora` | off | Include video/audio embedding connector linear layers |
| `--unsupported_tensors report|skip|error|sidecar` | `report` | Handling for changed tensors that regular LoRA cannot represent |
| `--dry_run` | off | Inspect key coverage and write the report without saving a LoRA |
| `--max_modules` | `0` | Optional extraction limit; `0` means no limit |

Pure LoRA cannot encode changed 1D tensors such as attention norm vectors, biases, or scale vectors. If a full fine-tune changed tensors like `q_norm.weight` or `k_norm.weight`, the extractor reports them instead of silently dropping them. Use `--unsupported_tensors error` when you want extraction to fail on those changes. `sidecar` writes exact delta tensors into the file for inspection, but standard LoRA loading ignores those sidecar tensors.

For connector-heavy LTX-2.3 full fine-tunes, run a dry pass first:

```bash
python -m musubi_tuner.ltx2_extract_lora ^
  --base_model /models/ltx-2.3-22b-dev.safetensors ^
  --finetuned_model /models/my_full_finetune.safetensors ^
  --save_to output/extracted_lora.safetensors ^
  --target_preset full ^
  --connector_lora ^
  --dry_run ^
  --report_json output/extract_coverage.json
```

If the report shows large unsupported norm or bias deltas, a higher-rank LoRA alone may still fail to match the full fine-tuned model. Try `--extract_mode dora`, a higher `--max_rank`, and `--connector_lora`, then compare samples against the full checkpoint.

Checkpoint rotation (`--save_last_n_epochs`) cleans up old ComfyUI checkpoints alongside originals. HuggingFace upload (`--huggingface_repo_id`) uploads both formats by default. Use `--no_save_original_lora` to upload only the ComfyUI checkpoint.

### Resuming Training
<sub>[↑ contents](#table-of-contents)</sub>

Requires `--save_state` to be enabled. By default, `--save_state_mode full` writes optimizer, scheduler, dataloader, RNG, and model/adapter state so the directory can be loaded with `--resume`. See the [Advanced Configuration guide](./advanced_config.md) for general `--save_state` / `--resume` behavior shared across all architectures.

| Flag | Description |
|------|-------------|
| `--resume <path>` | Resume from a specific state directory |
| `--autoresume` | Automatically resume from the latest state in `output_dir`. Ignored if `--resume` is specified. Starts from scratch if no state is found |
| `--save_state_mode full|minimal` | `full` saves resumable optimizer state. `minimal` saves a compact state artifact without optimizer/scheduler/dataloader state |
| `--reset_optimizer` | Clear optimizer momentum/variance on resume, keep model weights only |
| `--reset_optimizer_params` | Reset optimizer param groups (lr, weight_decay, etc.) to current CLI values on resume, keep momentum/variance |
| `--reset_dataloader` | Skip mid-epoch batch skip, restart epoch from beginning |

**Changing learning rate on resume:** When you resume from a saved state, the optimizer's learning rate is restored from the checkpoint — any new `--learning_rate` value on the command line is silently ignored. To apply a new learning rate, add `--reset_optimizer_params`. This resets lr, weight_decay, and other optimizer param-group settings to your current CLI values while keeping the accumulated momentum/variance intact.

Mid-epoch checkpoints record `step_in_epoch` in `resume_metadata.json`. On resume, already-processed batches are skipped to keep global step consistent. `--reset_dataloader` disables this.

The moving average loss is saved in state checkpoints and restored on resume.

`--save_state_mode minimal` is intended for compact saved artifacts when you do not need optimizer-continuation resume. Minimal state directories still include resume metadata and a manifest, but they are not loadable with `--resume`; `--autoresume` ignores them. To continue from a minimal artifact, load the saved adapter with `--network_weights` and start a fresh optimizer, or save with `--save_state_mode full`.

Newly saved state directories also include `state_manifest.json`, written only after the accelerator state save completes. `--autoresume` and the dashboard resume detector ignore incomplete state directories, so a crashed or force-killed save is not selected accidentally.

When training is launched from the dashboard, pressing Stop first requests a graceful interrupt save. The trainer writes an `*-interrupt-step########-state` directory with resume metadata, then exits. Pressing Stop again while the process is already stopping requests a force stop and skips the interrupt save.

---

## Merge LoRA into Base Model
<sub>[↑ contents](#table-of-contents)</sub>

**Script:** `ltx2_merge_lora_to_model.py`

```bash
python ltx2_merge_lora_to_model.py ^
  --dit base_model.safetensors ^
  --lora_weight lora.safetensors ^
  --save_merged_model merged_model.safetensors
```

To write a full checkpoint for a packaged base model, add `--save_full_model`:

```bash
python ltx2_merge_lora_to_model.py ^
  --dit base_model.safetensors ^
  --lora_weight lora_A.safetensors lora_B.safetensors ^
  --lora_multiplier 1.0 1.0 ^
  --save_merged_model merged_full_model.safetensors ^
  --save_full_model
```

### Merge-to-Base Arguments
<sub>[↑ contents](#table-of-contents)</sub>

- `--dit`: LTX-2 base model checkpoint (required).
- `--lora_weight`: One or more LoRA paths to merge sequentially (required).
- `--lora_multiplier`: Per-LoRA multipliers (default: all 1.0).
- `--save_merged_model`: Output merged model path (required).
- `--device cpu|cuda`: Device for merge computation (default: cuda if a CUDA device is available, otherwise cpu). Pass `--device cpu` to run on system RAM if you don't have enough VRAM.
- `--audio_video`: Load as audio-video model (for LTXAV checkpoints).
- `--audio_only`: Load as audio-only model.
- `--save_full_model`: Save a full checkpoint by replacing transformer weights in `--dit` and copying all other tensors from the original checkpoint. If `--dit` uses ComfyUI-style `model.diffusion_model.*` keys, that layout is preserved.

### Merge-to-Base Notes
<sub>[↑ contents](#table-of-contents)</sub>

- The output contains only transformer weights (VAE, vocoder, and text encoder are loaded separately by training/inference scripts).
- Original checkpoint metadata is preserved, so the merged file is directly usable with `--ltx2_checkpoint`.
- With `--save_full_model`, the output also contains non-transformer tensors copied from `--dit`. This is intended for packaged checkpoints where the base file includes components such as VAE, vocoder, or text encoder tensors.
- FP8 base models cannot be merged directly — merge into the bf16 base, then use `--fp8_base` at training time for on-the-fly quantization.

---

## Merge LTX-2 LoRAs
<sub>[↑ contents](#table-of-contents)</sub>

Use the dedicated LTX-2 LoRA merger to combine multiple LoRA files into a single LoRA checkpoint.

**Script:** `ltx2_merge_lora.py`

### Example Command (Windows)
<sub>[↑ contents](#table-of-contents)</sub>

```bash
python ltx2_merge_lora.py ^
  --lora_weight path/to/lora_A.safetensors path/to/lora_B.safetensors ^
  --lora_multiplier 1.0 1.0 ^
  --save_merged_lora path/to/merged_lora.safetensors
```

When merging multiple LoRAs that overlap on certain modules and only the first input's weights should win for the matched modules, pass a regex to `--preserve_first_match_pattern`:
```bash
python ltx2_merge_lora.py ^
  --lora_weight path/to/first.safetensors path/to/second.safetensors ^
  --preserve_first_match_pattern video_to_audio_attn ^
  --save_merged_lora path/to/merged_lora.safetensors
```

### LoRA Merge Arguments
<sub>[↑ contents](#table-of-contents)</sub>

- `--lora_weight`: Input LoRA paths to merge in order (required).
- `--lora_multiplier`: Per-LoRA multipliers aligned with `--lora_weight`. Use one value to apply the same multiplier to all inputs.
- `--save_merged_lora`: Output merged LoRA path (required).
- `--merge_method concat|orthogonal`: Merge method (default: `concat`). `concat` keeps all ranks by concatenation. `orthogonal` uses SVD refactorization to merge exactly 2 LoRAs with orthogonal projection.
- `--orthogonal_k_fraction`: Fraction of top singular directions projected out bilaterally before combining (default: `0.5`, range `[0, 1]`). Only used with `--merge_method orthogonal`.
- `--orthogonal_rank_mode sum|max|min`: Target rank mode for orthogonal merge (default: `sum`).
- `--preserve_first_match_pattern`: Regex matched against LoRA module prefixes. Matching modules keep only the first input LoRA that contains that module; later LoRAs are ignored for those modules.
- `--dtype auto|float32|float16|bfloat16`: Output tensor dtype. `auto` promotes from input dtypes.
- `--emit_alpha`: Force writing `<module>.alpha` keys in output.

### LoRA Merge Notes
<sub>[↑ contents](#table-of-contents)</sub>

- This merger is intended for LTX-2 LoRA formats used in this repo, including Comfy-style `lora_A/lora_B` weights.
- It handles different ranks and partial module overlap across input LoRAs.
- Orthogonal merge requires exactly 2 input LoRAs.

---

## Validation Datasets
<sub>[↑ contents](#table-of-contents)</sub>

> [!NOTE]
> Validation datasets are an extension specific to this LTX-2 trainer.

You can configure a separate validation dataset to track validation loss (`val_loss`) during training. This helps detect overfitting and compare training runs. Validation datasets use **exactly the same schema** as training datasets — any format that works for `[[datasets]]` works for `[[validation_datasets]]`.

### Configuration
<sub>[↑ contents](#table-of-contents)</sub>

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

Use a separate `cache_directory` for validation data to avoid mixing training and validation cache files.

### Caching
<sub>[↑ contents](#table-of-contents)</sub>

Validation datasets are automatically picked up by the caching scripts — no extra flags needed. Run the same caching commands you use for training:

```bash
python ltx2_cache_latents.py --dataset_config dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors --ltx2_mode av ...
python ltx2_cache_text_encoder_outputs.py --dataset_config dataset.toml --ltx2_checkpoint /path/to/ltx-2.safetensors --ltx2_mode av ...
```

Both scripts detect the `[[validation_datasets]]` section and cache latents/text embeddings for validation data alongside training data.

### Training Arguments
<sub>[↑ contents](#table-of-contents)</sub>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--validate_every_n_steps` | int | None | Run validation every N training steps |
| `--validate_every_n_epochs` | int | None | Run validation every N epochs |
| `--offload_optimizer_during_validation` | flag | off | Temporarily move CUDA optimizer state to CPU while validation/sample previews run |

At least one of these must be set for validation to run. If neither is set, validation is skipped even if `[[validation_datasets]]` is configured.

### Example
<sub>[↑ contents](#table-of-contents)</sub>

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --dataset_config dataset.toml ^
  --validate_every_n_steps 100 ^
  ... (other training args)
```

### How It Works
<sub>[↑ contents](#table-of-contents)</sub>

1. A separate validation dataloader is created with `batch_size=1` and `shuffle=False` (deterministic order).
2. At the configured interval, the model switches to eval mode and runs inference on all validation samples with `torch.no_grad()`.
3. The average validation loss is computed with the active `--loss_type` and logged as `val_loss` to TensorBoard/WandB.
4. The model is restored to training mode and training continues.

### Tips
<sub>[↑ contents](#table-of-contents)</sub>

- **Keep validation sets small.** Aim for 5-20% of your main dataset size. Validation runs on every sample each time, so 10-50 clips is usually enough. Large validation sets slow down training.
- **Use held-out data.** Validation data should be different from the training set for meaningful overfitting detection. In extreme cases, using a small subset of the training data is acceptable — it will still help catch divergence, but won't reliably detect overfitting.
- **Monitor the gap.** If `val_loss` starts increasing while training loss keeps decreasing, you're overfitting — consider stopping or reducing the learning rate.
- **Same preprocessing.** Validation data goes through the same frame extraction, FPS resampling, and resolution bucketing as training data.

---

## Directory Structure
<sub>[↑ contents](#table-of-contents)</sub>

### Raw Dataset Layout (Example)
<sub>[↑ contents](#table-of-contents)</sub>

```
dataset_root/
  videos/
    000001.mp4
    000001.txt       # caption
    000002.mp4
    000002.txt
```

### Cache Directory Layout (After Caching)
<sub>[↑ contents](#table-of-contents)</sub>

```
cache_directory/
  000001_1024x0576_ltx2.safetensors       # video latents
  000001_ltx2_te.safetensors              # text encoder outputs
  000001_1024x0576_ltx2_audio.safetensors # audio latents (av mode only)
  000001_1024x0576_ltx2_dino.safetensors  # DINOv2 features (CREPA dino mode only)

reference_cache_directory/                  # IC-LoRA only
  000001_1024x0576_ltx2.safetensors       # reference latents
```

---

## Troubleshooting
<sub>[↑ contents](#table-of-contents)</sub>

| Error | Cause | Solution |
|-------|-------|----------|
| `mv_act` / `loss/video_mask_active` appears even though no `loss_mask_directory` is configured | Usually first-frame conditioning, not dataset masked loss. `--ltx2_first_frame_conditioning_p` defaults to `0.1`; when it fires, frame 0 is clean conditioning and is excluded from loss. | No fix needed if you want the default occasional I2V conditioning. For pure T2V and no first-frame-related mask metrics, pass `--ltx2_first_frame_conditioning_p 0.0`. |
| Need to inspect LTX-2 dataset parsing/buckets without starting training | Dataset config may be valid or invalid, but a full training launch is expensive. | Run the LTX-2 training command with `--debug_dataset`. It builds the dataset, prints the dataset summary, validates buckets, and exits before model setup. This is an LTX-2 CLI flag, not a dataset TOML key. |
| Missing cache keys during training | Caching incomplete | Run both `ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py` |
| FlashAttention varlen mask-length mismatch (for example `expects mask length 1920, got 1024`) after checkpoint switch | Stale text cache from a different checkpoint/version/mode | Re-run `ltx2_cache_text_encoder_outputs.py` with the same `--ltx2_checkpoint` and `--ltx2_mode` as training. Remove old `*_ltx2_te.safetensors` if needed. |
| Samples become progressively noisier/degraded when training with `--flash_attn` | FlashAttention install/runtime mismatch (CUDA/PyTorch/flash-attn build) | Switch to `--sdpa` to confirm baseline stability. If SDPA is stable, reinstall FlashAttention for your exact CUDA + PyTorch versions and retry. |
| Training fails after changing `--ltx2_checkpoint` even though args look correct | Reused latent/text caches generated from a different checkpoint | Re-run both caches (`ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py`) and regenerate `--dataset_manifest` before training. |
| OOM appears after removing `--fp8_base` while using an FP8 checkpoint | Base model no longer uses FP8 loading path, so VRAM increases sharply | Keep `--fp8_base` enabled for FP8 checkpoints (typically with `--mixed_precision bf16`) |
| Error when combining `--fp8_scaled` with an FP8 checkpoint (old behavior) | Checkpoint has FP8 weights without `weight_scale` keys (non-standard format) | Use a standard bf16 checkpoint, or a properly exported FP8 checkpoint that includes scale tensors |
| Missing `*_ltx2_audio.safetensors` | Audio caching skipped | Re-run latent caching with `--ltx2_mode av` |
| Gemma connector weights missing | Incorrect checkpoint | Ensure `--ltx2_checkpoint` (or `--ltx2_text_encoder_checkpoint`) contains Gemma connector weights |
| Gemma OOM | Model too large | Use `--gemma_load_in_8bit` or `--gemma_load_in_4bit` with `--device cuda`, or use `--gemma_safetensors` with an FP8 file |
| Startup OOM with `--dop` or `--blank_preservation` | Live preservation-prompt encoding briefly loads Gemma alongside the model; DOP/blank preservation do not need Gemma after startup. | Precache with `--precache_preservation_prompts`, then train with `--use_precached_preservation`. |
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
| `ValueError: Gemma safetensors is missing required language-model tensors` with `missing_buffers` mentioning `full_attention_inv_freq` or `sliding_attention_inv_freq` | `transformers>=5.0` renamed Gemma3 rotary embedding buffers (`rotary_emb.inv_freq` → `rotary_emb.full_attention_inv_freq` / `sliding_attention_inv_freq`). The derivable-buffer suffix check expects `.inv_freq` and does not match the new `_inv_freq` suffix. The safetensors file is correct — rotary buffers are non-persistent and computed from config at init time. | `pip install transformers==4.57.6` (pinned in `pyproject.toml`), or reinstall all deps with `pip install -e .` |
| A nominally video-only LoRA contains audio or AV cross-modal adapter keys | `--ltx2_mode av` constructs the audio/cross-modal modules, and a broad target preset can select them. In `--ltx2_mode v`/`video`, those modules are not constructed. | Use `--ltx2_mode v` when no audio path is needed. If AV mode is required, use a `video_*` preset (`video_sa`, `video_sa_ff`, or `video_sa_ca_ff`) to restrict adapter targets to the video branch. See [LoRA Targets](#lora-targets). |
| `loss_a` too low but `loss_v` still high (audio overfitting) | Audio latent space converges faster than video; audio gradients dominate shared weights | Lower `--audio_loss_weight` (e.g., 0.3), or use `--audio_loss_balance_mode ema_mag` to auto-dampen audio when it exceeds `target_ratio × video_loss`. Reduce audio learning rate with `--audio_lr 1e-6` or fine-grained `--lr_args audio_attn=1e-6 audio_ff=1e-6`. Disable `--audio_dop` / `--audio_silence_regularizer` if active — they add more audio signal. |
| `loss_a` absent or not dropping in mixed dataset (audio starvation) | Audio batches too rare — non-audio steps outnumber audio steps, audio branch gets insufficient supervision | Increase `num_repeats` on audio datasets (target 30-50% audio steps). Add `--audio_loss_balance_mode inv_freq` to auto-boost audio weight. Use `--audio_dop` or `--audio_silence_regularizer` to provide audio signal on non-audio steps. Check caching summary for `failed > 0`. |

### Mixed Audio-Video Training
<sub>[↑ contents](#table-of-contents)</sub>

Use this section for joint AV LoRA (`--ltx2_mode av`) when the dataset mixes audio-bearing clips with silent, video-only, image, or reference-conditioned samples. The common failure modes are audio starvation (`loss_a` is absent or rare) and audio overfitting (`loss_a` drops much faster than `loss_v` while video, identity, or motion quality drifts). These controls are training-only; the saved LoRA loads at inference.

**1. Keep audio batches visible.** Use dataset `num_repeats` so audio-bearing clips are not outnumbered by non-audio samples:
```toml
[[datasets]]
video_directory = "audio_video_clips"
num_repeats = 5
```

The audio-aware sampler can also be used when both audio and non-audio batches are available:
```bash
--audio_batch_probability 0.4
```
With gradient accumulation, prefer a hard quota:
```bash
--gradient_accumulation_steps 4 --min_audio_batches_per_accum 1
```
Do not combine `--audio_batch_probability` / `--min_audio_batches_per_accum` with `--accumulation_group_by`; both need to own DataLoader sampling. See [Audio-Video Support](#audio-video-support).

**2. Separate incompatible batches.** Use `--separate_audio_buckets` when audio and non-audio items share a dataset at `batch_size > 1`. This avoids text-embedding shape mismatches and keeps non-audio batches cheaper.

**3. Lower audio learning rate.** Audio modules may need a lower LR than video modules in mixed AV runs. Use `--audio_lr` for a broad audio-side override or `--lr_args` for module-level control:
```bash
--learning_rate 1e-4 --audio_lr 3e-5
```

**4. Lower audio LoRA rank.** Use `--audio_dim` / `--audio_alpha` when audio modules should have less adaptation capacity than video modules:
```bash
--network_dim 32 --audio_dim 8 --audio_alpha 8
```

**5. Balance audio/video losses.** `--audio_loss_balance_mode` controls dynamic audio loss weighting:
- `inv_freq`: scales audio by inverse audio-batch frequency EMA; useful when audio batches are rare.
- `ema_mag`: tracks audio/video loss EMAs and scales audio toward `--audio_loss_balance_target_ratio`; can boost or dampen audio.
- `uncertainty`: learns two log-variance scalars jointly with LoRA parameters.
- `ogm_ge`: attenuates the lower-loss / faster-learning modality on each AV step.

Examples:
```bash
--audio_loss_balance_mode inv_freq --audio_loss_balance_min 0.05 --audio_loss_balance_max 4.0
```
```bash
--audio_loss_balance_mode ema_mag --audio_loss_balance_target_ratio 0.33
```

For audio-starved runs, combine `inv_freq` with the supervision monitor:
```bash
--audio_loss_balance_mode inv_freq ^
--audio_supervision_mode warn ^
--audio_supervision_min_ratio 0.5
```

When audio overfits before video, combine a lower audio LR/rank with `ema_mag`:
```bash
--learning_rate 1e-4 --audio_lr 3e-5 ^
--network_dim 32 --audio_dim 8 --audio_alpha 8 ^
--audio_loss_balance_mode ema_mag ^
--audio_loss_balance_target_ratio 0.25
```

**6. Add signal on non-audio steps only when needed.** `--audio_dop` preserves base audio predictions on non-audio batches. It runs LoRA-off and LoRA-on forwards on synthetic silence audio latents, logs `loss/audio_dop`, and has no cost on batches that already contain audio:
```bash
--audio_dop --audio_dop_args multiplier=0.5
```

`--audio_silence_regularizer` instead trains missing-audio batches toward silence. It is cheaper than Audio DOP, but silence is a training target rather than a preservation reference:
```bash
--audio_silence_regularizer --audio_silence_regularizer_weight 0.5
```

`--audio_dop` and `--audio_silence_regularizer` should not be combined; if both are set, the trainer warns (it does not error) and you should pick one. Prefer checking sampler balance and `loss_a` first, because both add extra audio supervision.

**7. Drop modality text conditioning independently.** Per-modality caption dropout can regularize AV conditioning and train mixed conditional states:
```bash
--video_caption_dropout_rate 0.05 --audio_caption_dropout_rate 0.10
```

This can be combined with the audio-aware sampler, `--audio_loss_balance_mode ema_mag`, and `--dcr`.

**8. Freeze the faster modality when it dominates.** Modality freezing toggles audio/video LoRA parameters based on the audio/video loss EMA ratio:
```bash
--modality_freeze_check_interval 500 ^
--modality_freeze_ratio_threshold 0.5 ^
--modality_freeze_warmup_steps 200
```

Logged metrics include `modality_freeze/state`, `modality_freeze/video_loss_ema`, and `modality_freeze/audio_loss_ema`.

**9. Use Self-Flow when representation alignment is part of the run.** `--self_flow` can add video or audio representation alignment in video, AV, and audio-only modes. Keep its detailed setup in the [Self-Flow](#self-flow-self-supervised-flow-matching) section, and only enable audio alignment when you intentionally want that objective:
```bash
--self_flow --self_flow_args lambda_self_flow=0.1 lambda_audio=0.1 teacher_mode=ema
```

**10. Route cross-modal gradients deliberately for `av_ic` and sync-sensitive runs.** `--dcr --dcr_args reference_detach=true` can be combined with the audio-aware sampler, `ema_mag`, and per-modality caption dropout. Add `--tarp` or Cross-Task Synergy only when AV sync or cross-modal alignment is the target and the compute cost is acceptable:
```bash
--dcr --dcr_args reference_detach=true
```
```bash
--cts_lambda_video_driven 0.3 --cts_lambda_audio_driven 0.1
```

**11. Override warmup by optimizer group when needed.** Per-group warmup keeps the same scheduler family but can stretch warmup differently for audio and video groups:
```bash
--lr_group_warmup_args audio=500 video=1500
```

**12. Verify caching and signal before scaling.**
- If `failed > 0` in latent caching summary, audio extraction is broken for those items.
- After switching from video-only to AV mode, re-run both latent and text encoder caching without `--skip_existing`.
- `loss_a` dropping means audio is learning; absent/zero usually means no audio batches are forming; degradation over time can indicate forgetting.
- Track `grad_norm/video`, `grad_norm/audio`, and `grad_norm/audio_video_ratio` during AV runs.

### AV Loss Curricula
<sub>[↑ contents](#table-of-contents)</sub>

`--av_curriculum_mode` is an opt-in alternative to flat joint AV loss weighting.
It changes which primary diffusion loss contributes gradients while both
modality streams still run in every forward:

- `alternating`: train video for K successful optimizer steps, then audio for K
  steps, and repeat;
- `two_stage`: use one `video`, `audio`, or `joint` policy for a fixed number of
  successful optimizer steps, then switch to another policy;
- `none` (default): preserve ordinary flat joint AV training.

The phase is derived from the optimizer `global_step`. All microbatches in one
gradient-accumulation window therefore use the same policy. A skipped optimizer
update does not advance the phase, and checkpoint resume reconstructs the same
phase without separate curriculum state. Validation always evaluates the joint
video+audio objective.

Alternating example:

```bash
--ltx2_mode av ^
--av_curriculum_mode alternating ^
--av_curriculum_interval_steps 100 ^
--av_curriculum_start_modality video
```

Two-stage video warm-up followed by joint training:

```bash
--ltx2_mode av ^
--av_curriculum_mode two_stage ^
--av_curriculum_stage1_steps 1000 ^
--av_curriculum_stage1_policy video ^
--av_curriculum_stage2_policy joint
```

Requirements:

- every training batch must contain paired video and audio;
- `--video_loss_weight` and `--audio_loss_weight` must both be positive;
- use joint training, no IC-LoRA strategy, and
  `--audio_loss_balance_mode none`;
- do not combine the curriculum with the modality freezer, synthetic-silence
  fallback, CREPA, Self-Flow, or Cross-Task Synergy auxiliary losses.

Static video/audio loss weights remain the base weights for whichever policy is
active. Tracker metrics include the current phase, effective weights, raw
video/audio losses, and the existing per-branch gradient norms under
`loss/av_curriculum/*`. `policy_id` is `0` for joint, `1` for video, and `2`
for audio.

### Technical Notes
<sub>[↑ contents](#table-of-contents)</sub>

- **Float32 AdaLN**: The transformer applies Adaptive Layer Norm (AdaLN) shift/scale operations in float32, then casts back to the working dtype. This prevents overflow that can occur when bf16 scale values multiply bf16 hidden states. The fix is always active and requires no flags.
- **Loss dtype**: The LTX-2 training path casts the model prediction to the working dtype, but the task loss itself (MSE, L1, Huber) is computed in float32 — both prediction and target are upcast inside the reduction for numerical stability. Internal regularization losses (motion preservation, CREPA, Self-Flow, latent temporal objectives) always use their own configured loss and are unaffected by global `--loss_type`.

For additional troubleshooting resources, see the [LTX-2 documentation hub](https://docs.ltx.video/open-source-model/getting-started/overview), the [Banodoco Discord](https://discord.gg/banodoco) community, and the [awesome-ltx2](https://github.com/wildminder/awesome-ltx2) curated resource list.

---

## Windows Setup / Update Script
<sub>[↑ contents](#table-of-contents)</sub>

[`scripts/install.ps1`](https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2/scripts/install.ps1) is the Windows setup and maintenance entry point.

> [!WARNING]
> The setup script, dashboard, and GUI are **experimental**. Their behavior, layout, and generated files may change between versions. Unsupported local environments may still need manual installation steps. Not all training and inference paths are wired through the dashboard — some advanced flags are CLI-only.

The script can run these actions, depending on the selected options and current machine state:

- install or locate `git`, Python, and Node.js/npm
- clone the repository, update an existing checkout, or switch the target branch with `-Branch`
- create the repo-local `venv` or reuse an existing `venv\Scripts\python.exe`
- install PyTorch for `cu124`, `cu128`, `cu130`, or `cpu`, then install the project with dashboard extras
- build the dashboard frontend with `npm install` and `npm run build`
- write `launch_musubi_dashboard.cmd` and `launch_musubi_setup.cmd`
- optionally create desktop shortcuts for the dashboard and setup/update launcher
- write `.musubi_install_state.json` for later setup/status checks
- optionally start the dashboard launcher at the end of the run

**Interactive one-liner:**

```powershell
irm https://raw.githubusercontent.com/AkaneTendo25/musubi-tuner/ltx-2/scripts/install.ps1 | iex
```

With no parameters, the script defaults to branch `ltx-2`, CUDA `cu128`, Python `3.12`, dashboard host `127.0.0.1`, and port `7860`. Interactive mode prints the detected environment and lets you choose which actions to run.

**Saved script with explicit parameters:**

```powershell
irm https://raw.githubusercontent.com/AkaneTendo25/musubi-tuner/ltx-2/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Cuda cu124 -PythonVersion 3.11 -NonInteractive
```

Available parameters: `-InstallRoot`, `-RepoUrl`, `-Branch`, `-RepoDir`, `-Cuda` (`cu124`/`cu128`/`cu130`/`cpu`), `-PythonVersion` (`3.10`/`3.11`/`3.12`/`3.13`), `-Port`, `-DashboardHost`, `-NonInteractive`, `-StrictPreflight`, and `-PreflightOnly`.

Use `-PreflightOnly` to run the environment checks without making install changes. The script writes a timestamped log to `%TEMP%\musubi_ltx2_install_*.log`; on failure it prints a support bundle with the current step, exception details, and log path.

### Dashboard Usage
<sub>[↑ contents](#table-of-contents)</sub>

> [!NOTE]
> The dashboard GUI is experimental. Common training, caching, and inference flows are wired up; some advanced flags remain CLI-only. UI layout, validation messages, and dashboard metrics may change between versions.

After the installer finishes, start the dashboard from the generated `Musubi Tuner Dashboard` desktop shortcut or from `launch_musubi_dashboard.cmd` in the repository directory.

You can also start it manually from the repo-local virtual environment:

```powershell
venv\Scripts\python.exe -m musubi_tuner.gui_dashboard --host 127.0.0.1 --port 7860
```

Open the printed browser URL, usually `http://127.0.0.1:7860/`.

Basic flow:

- Use `Overview` (the home tab) to create or load a `project.json`.
- Use `Dataset` to define training and validation datasets.
- Use `Caching` to run latent and text encoder cache jobs.
- Use `Training` to configure LoRA training and start/stop the training process.
- Use `Samples` for training sample prompts.
- Use `Inference` to run inference from the selected project settings.
- Use the `Manage` tab (settings) to view install status, launcher/shortcut status, and repository status, and to open the setup/update tool.

When a cache, training, slider training, or inference job is started from the dashboard, the dashboard shows process status and logs. For training jobs started from the dashboard, the dashboard also reads the training metrics written by the trainer.

---

# Part III — Advanced

## Advanced Training
<sub>[↑ contents](#table-of-contents)</sub>

All training arguments can be placed in a `.toml` config file instead of on the command line via `--config_file config.toml`. See the [configuration files guide](./advanced_config.md) for format details.

### Forward Explorative Modeling (XM)
<sub>[↑ contents](#table-of-contents)</sub>

`--ltx2_xm_k K` enables the hard-min Forward Explorative Modeling objective described in
[Explorative Modeling](https://arxiv.org/abs/2607.27372). This is a hybrid use of Forward XM inside the existing LTX
flow-matching objective. For each sample, the implementation:

1. keeps the dataset sample, timestep, and conditioning RNG state fixed;
2. evaluates `K` different video-noise candidates without gradients;
3. selects the candidate with the lowest configured masked reconstruction score for that sample; and
4. runs the selected candidate once more with gradients.

Candidate 0 uses the noise already sampled by ordinary training. The remaining candidates use independent private
noise draws. In AV training, every candidate also has its own audio-noise draw, and the weighted sum of the video and
audio reconstruction scores selects one joint video/audio candidate per sample. The selected video and audio noise
tensors are passed directly to the gradient-bearing forward.

XM changes training only. Saved LoRAs and full-finetune checkpoints use the ordinary LTX inference path. `K=1` is the
default and takes the ordinary training path without XM candidate generation, RNG snapshots, or XM metrics. Valid
values are from `1` through `32`. To enable two-candidate selection:

```bash
--ltx2_xm_k 2
```

With `K>1`, each step executes `K` no-gradient scoring forwards followed by one gradient-bearing forward for the
selected candidate. Candidate activation graphs are not retained. The large activation-memory term therefore does not
grow with `K`; small seed, score, index, and selected-noise state is retained separately. Training compute grows with
the number of scoring forwards.

Every XM candidate starts from the same captured Python, CPU, and CUDA conditioning RNG state. The implementation
checks whether all candidates and the gradient-bearing winner replay consume the same RNG stream, while the selected
video and audio noise tensors are replayed directly. The LoRA/network training loop writes the following TensorBoard
metrics:

| Metric | Definition |
| --- | --- |
| `loss/xm/k` | Configured candidate count. |
| `loss/xm/candidate0_score` | Mean reconstruction score for the ordinary-training noise candidate. |
| `loss/xm/winner_score` | Mean lowest candidate score selected without gradients. |
| `loss/xm/selection_gain` | `candidate0_score - winner_score`; it is non-negative for finite scores by construction. |
| `loss/xm/winner_index_mean` | Mean selected candidate index over the batch. |
| `loss/xm/winner_<i>_fraction` | Fraction of samples won by candidate `i`. Fractions sum to 1 for a valid step. |
| `loss/xm/invalid_samples` | Number of samples without a finite selected score. This is `0` on a logged valid step; the step raises before backward if any sample has no finite candidate. |
| `loss/xm/condition_rng_verified` | `1` if all scoring forwards and the winner forward end at the same exact Python/CPU/CUDA conditioning RNG state; otherwise `0`. A zero value is reported but does not stop training. |
| `loss/xm/replay_max_abs_error` | Maximum absolute difference between the selected no-gradient score and the gradient-bearing replay score. |
| `loss/xm/replay_mean_abs_error` | Mean absolute difference between those two scores. |

The replay-error metrics are numerical diagnostics, not RNG checks and not stopping criteria. No-gradient and
gradient-bearing mixed-precision forwards can produce different scores even when the selected noise and conditioning
RNG state are unchanged. XM metrics describe candidate selection and replay; they do not measure generated-output
quality. The full-finetune loop uses the same XM selection path and stores `ss_ltx2_xm_k` in checkpoint metadata, but
does not currently export these per-step XM values to TensorBoard.

Because XM uses a hard minimum, increasing `K` can lower the training score mechanically. Training losses from
different `K` values are therefore not directly comparable. For an experiment, report `K`, optimizer updates, and
training compute or wall time; compare held-out generations at either matched updates or matched compute and state
which comparison is used.

`K>1` currently supports single-process training only. It is not supported with `--self_flow`, `--hfato`,
`--differential_guidance`, `--crepa`, `--tread`, `--latent_temporal_weighting`, `--latent_delta_loss`,
`--av_attention_loss_weighting`, `--blank_preservation`, `--dop`, `--prior_divergence`, `--audio_dop`,
`--motion_preservation`, `--motion_attention_preservation`, `--image_prior_ft`, non-`none`
`--audio_loss_balance_mode`, nonzero Cross-Task Synergy lambdas, `--ltx2_fsdp`, `--ltx2_model_parallel`, or
`--ltx2_remote_stage`.

For full fine-tuning with `K>1`, do not combine dataset-level `video_loss_weight` or `audio_loss_weight` values with
non-default CLI `--video_loss_weight` or `--audio_loss_weight` values. Candidate selection sees the dataset-level
weights, while the full-finetune loss applies the additional CLI multiplier after candidate selection.

### Optional: Source-Free Training from Cache
<sub>[↑ contents](#table-of-contents)</sub>

If you cached with `--save_dataset_manifest`, you can train without source dataset paths:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --dataset_manifest dataset_manifest.json ^
  ... (other training args)
```

Use `--dataset_manifest` instead of `--dataset_config`.

Dashboard workflow: the Project page has a **Cache & Start Training** action. It runs latent caching, waits for success, runs text caching, waits for success, and then starts training. The Project page shows the active stage, shared progress, a stop button for the running stage, and links to open the relevant caching or training view. If no dataset manifest path is set, the action uses `dataset_manifest.json` in the project folder and sets both caching and training to that manifest before it starts.

### DoRA LoRA Training
<sub>[↑ contents](#table-of-contents)</sub>

DoRA adds a separate learnable magnitude vector to the standard LTX-2 LoRA backend. It is opt-in and uses the same target presets, rank, alpha, optimizer, and dataset settings as regular LoRA. (Credit: [Ada123-a](https://github.com/Ada123-a) — [PR #80](https://github.com/AkaneTendo25/musubi-tuner/pull/80))

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as standard LoRA) ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 ^
  --network_alpha 16 ^
  --network_args "use_dora=true" ^
  --output_name ltx2_dora
```

Training-time ComfyUI export is supported for DoRA LoRA. The native Musubi checkpoint stores `lora_magnitude_vector.weight`; the generated `*.comfy.safetensors` file stores the equivalent ComfyUI `dora_scale` tensors.

Set `dora_scale_lr_ratio=N` in `--network_args` to train the DoRA/DoKr magnitude parameters at `N` times the adapter learning rate. This applies to native DoRA, DoKr, DoRA-OFT, and DoKr-OFT; `0` keeps the magnitude parameters out of the optimizer. When omitted, magnitude and direction parameters use the same learning rate.

### Rank-Stabilized LoRA (rsLoRA)
<sub>[↑ contents](#table-of-contents)</sub>

rsLoRA keeps the standard LoRA tensor layout but changes the adapter scale from `alpha / rank` to `alpha / sqrt(rank)`. This is useful for higher-rank LoRA runs where the regular `alpha / rank` scale can become too small. It is opt-in and currently applies to the native LoRA/DoRA backend only; it is not supported with OFT, DoRA-OFT, DoKr-OFT, LyCORIS, LoHa, or LoKr.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as standard LoRA) ^
  --network_module networks.lora_ltx2 ^
  --network_dim 64 ^
  --network_alpha 32 ^
  --network_args "use_rslora=true" ^
  --output_name ltx2_rslora
```

When `use_rslora=true` is not supplied, training uses the standard `alpha / rank` LoRA scale. Exported rsLoRA adapters remain standard LoRA checkpoints: the saved `alpha` value is adjusted so regular LoRA loaders apply the same effective scale.

The same `use_dora=true` flag enables DoKr when the native LoKr backend is selected:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as standard LoRA) ^
  --network_module networks.lokr ^
  --network_dim 16 ^
  --network_alpha 16 ^
  --network_args "use_dora=true" ^
  --output_name ltx2_dokr
```

DoKr is also opt-in. Without `use_dora=true`, `networks.lokr` keeps the regular LoKr path.

DoRA-OFT and DoKr-OFT add an opt-in OFT rotation to the DoRA/DoKr magnitude path. Use `use_dora_oft=true` instead of `use_dora=true`; the two modes are mutually exclusive, and adaptive rank is not supported with DoRA-OFT/DoKr-OFT.

```bash
# Native LoRA + DoRA-OFT
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as standard LoRA) ^
  --network_module networks.lora_ltx2 ^
  --network_dim 32 ^
  --network_alpha 16 ^
  --network_args "use_dora_oft=true" ^
  --output_name ltx2_dora_oft

# Native LoKr + DoKr-OFT
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  ... (same args as standard LoRA) ^
  --network_module networks.lokr ^
  --network_dim 16 ^
  --network_alpha 16 ^
  --network_args "use_dora_oft=true" ^
  --output_name ltx2_dokr_oft
```

OFT (`use_oft=true`), DoRA-OFT, and DoKr-OFT default to a scaled rotation (`scaled_oft=true`, divides the skew by `2*sqrt(block_size-1)` before the Cayley transform). Pass `scaled_oft=false` for the canonical OFTv2 rotation with no divisor.

OFT / DoRA-OFT / DoKr-OFT weights use this trainer's own format (input-side block rotation, PEFT OFTv2 lineage) and are loadable only by this trainer. They are not interchangeable with kohya-ss / LyCORIS / diffusers diag-OFT checkpoints, which rotate output features under a different block structure; there is no lossless conversion between the two.

### Advanced: LyCORIS/LoKR Training
<sub>[↑ contents](#table-of-contents)</sub>

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
- Project config fields map to CLI as follows: `training.lycoris_algo` -> `algo`, `training.lycoris_factor` -> `factor`, `training.lycoris_conv_dim` -> `conv_dim`, `training.lycoris_conv_alpha` -> `conv_alpha`, and `training.lycoris_dropout` -> `dropout`.
- `training.lycoris_config` emits `--lycoris_config`; `training.init_lokr_norm` emits `--init_lokr_norm`; `training.lycoris_quantized_base_check_mode` emits `--lycoris_quantized_base_check_mode` when changed from the default `warn`.
- Supported algorithm values: `lora`, `loha`, `lokr`, `locon`, `ia3`, and `full`.
- `locon` uses the LoRA-style path plus convolution parameters (`conv_dim`, `conv_alpha`) when the selected targets include compatible convolutional layers. LTX-2 transformer training is mostly linear-layer focused, so LoCon is primarily exposed for LyCORIS compatibility.
- `ia3` is a tiny multiplicative adapter exposed through LyCORIS; it usually needs a higher learning rate than LoRA-style adapters.
- `full` is LyCORIS native full-matrix adapter mode, not the same as LTX-2 full-parameter fine-tuning.
- Use `--init_lokr_norm` only with LoKR (`algo=lokr`).
- If both TOML and `--network_args` are used, `--network_args` can override nested TOML keys with:
  - `modules.<name>.<param>=...`
  - `init.<param>=...`

`--lycoris_config` requires `--network_module lycoris.kohya`.

### Memory Optimization
<sub>[↑ contents](#table-of-contents)</sub>

For additional training and inference speedups, see the [torch.compile Support](./torch_compile.md) documentation.

#### Quantization Options
<sub>[↑ contents](#table-of-contents)</sub>

The following table restores the historical measurements documented for this branch. The repository does not include the raw output artifact used to produce them, so treat the values as reported reference data, not as a reproduced benchmark or capacity guarantee.

| Method | Base-weight VRAM (19B) | Weight Error (MAE) | SNR | Cosine Similarity |
|--------|------------------|--------------------|-----|-------------------|
| BF16 (baseline) | ~38 GB | 0.0011 | 55.6 dB | 0.999999 |
| `--fp8_base --fp8_scaled` | ~19 GB | 0.0171 (15x BF16) | 32.0 dB | 0.999686 |
| `--nf4_base` | ~10 GB | 0.0678 (60x BF16) | 21.2 dB | 0.996188 |
| `--nf4_base --loftq_init` | ~10 GB | 0.0654 (60x BF16) | 21.5 dB | 0.996437 |
| INT4 ConvRot weights (MSE-clipped g256) | ~10 GB | 0.0037 | 18.9 dB | 0.993353 |

The reported values used representative LTX-2 transformer shapes, except the INT4 ConvRot row, which was reported from a full LTX-2.3 transformer-block quality report. MAE is the mean absolute error between original and reconstructed weights. The LoftQ row includes the rank-32, two-iteration LoRA correction. The VRAM column describes base-weight storage, not peak training memory.

All quantized-base paths are opt-in; a run that omits their flags retains the normal base-loading path. Peak training memory also includes activations, temporary quantization buffers, adapter parameters, gradients, optimizer state, and optional sampling models. This repository does not guarantee output or training equivalence between a quantized path and bf16/fp16. Use the generated reconstruction report where available and compare task-specific samples before selecting a format. INT4 ConvRot is disabled unless one of the W4 mode flags is selected. In `--w4a8` mode, weights stay packed INT4 while forward activations and backward grad-output use row-wise INT8. LoftQ initializes LoRA weights from a quantization residual via SVD.

- `--fp8_base`: keep base model weights in FP8 path (~19 GB base weights).
- `--fp8_scaled`: quantize checkpoint weights to FP8 at load time. Works with both standard (bf16/fp16/fp32) and pre-quantized FP8 checkpoints (the latter are dequantized to bf16 first, then re-quantized).
- `--fp8_keep_blocks "0,1,2,45"`: with `--fp8_scaled`, keep selected transformer blocks in high precision instead of FP8 (comma lists and ranges such as `0-2,45`). By default all transformer blocks are quantized. For reference, the official `Lightricks/LTX-2.3-fp8` checkpoints leave their boundary blocks in bf16 (dev `0,1,46,47`, distilled `0,1,2,3,47`); replicating that here is optional and off by default. Ignored when `--fp8_scaled` is not set.
- `--fp8_w8a8`: with `--fp8_base --fp8_scaled`, use W8A8 activation quantization for LoRA training. `--w8a8_mode int8` is the default; `--w8a8_mode fp8` keeps FP8 weights and dequantizes transiently.
- `--w8a8_backend torch|triton|cutlass`: explicit backend selector for int8 W8A8 Linear matmuls. Omit it to preserve the default W8A8 backend resolution. `torch` uses `torch._int_mm` and the standard int32-to-compute-dtype scale/bias epilogue. `triton` uses the Triton int8 matmul and, when dtype/layout constraints are met, fuses row scale, column scale, bias, and output cast into the forward matmul epilogue. `cutlass` builds a native CUTLASS extension from headers supplied by `LTX2_CUTLASS_INCLUDE_DIR` and returns an int32 accumulator to the standard scale/bias epilogue.
- `--nf4_base`: NF4 4-bit quantization (~10 GB base weights). Mutually exclusive with `--fp8_base`. See [NF4 Quantization](#nf4-quantization) below.
- `--int8_base`: load a pre-quantized Optimum-Quanto `qint8` checkpoint and train a LoRA over the frozen int8 base. See [int8 Base (Optimum-Quanto)](#int8-base-optimum-quanto) below.
- `--int8_convrot_dynamic`: quantize a standard checkpoint to INT8 ConvRot at load time. See [INT8 ConvRot Base](#int8-convrot-base) below.
- `--w4a4g4` / `--w4a4g8` / `--w4a8`: local INT4 ConvRot modes (weights stored as packed signed INT4). `--w4a8` uses INT8 activations and grad-output, `--w4a4g8` uses INT4 activations with INT8 grad-output, and `--w4a4g4` uses INT4 for activations and grad-output. All three auto-detect a pre-quantized versus bf16/fp16 checkpoint. See [W4A4G4 Training](#w4a4g4-training) below.
- `--quantize_device cpu|cuda|gpu`: Device for startup quantization in NF4/FP8 and dynamic INT8/INT4 ConvRot paths (default: `cuda`). `cpu` loads and quantizes weights on CPU, then moves the quantized result to GPU; this avoids load-time GPU high-water marks at the cost of more host RAM and startup time. `cuda` loads and quantizes directly on GPU. For NF4/FP8 it overrides `LTX2_NF4_CALC_DEVICE` / `LTX2_FP8_CALC_DEVICE`.

#### Other Memory Options
<sub>[↑ contents](#table-of-contents)</sub>

| Argument | Description |
|----------|-------------|
| `--blocks_to_swap X` | Offload X transformer blocks to CPU (up to one less than the model's block count; e.g. 47 for the 48-block 22B checkpoint). Higher = more VRAM saved, more CPU↔GPU overhead |
| `--use_pinned_memory_for_block_swap` | Use pinned host memory for faster block transfers and checkpoint recomputation; recommended for classic block swap when host RAM permits |
| `--block_swap_h2d_only` | Stream frozen LoRA base blocks Host→Device only; requires `--blocks_to_swap X` and `--gradient_checkpointing` |
| `--block_swap_ring_size N` | H2D-only GPU ring size. `2` (default) uses double buffering; `1` allocates one streaming buffer and cannot overlap prefetch with compute |
| `--gradient_checkpointing` | Reduce VRAM by recomputing activations during backward pass |
| `--gradient_checkpointing_cpu_offload` | Offload activations to CPU during gradient checkpointing |
| `--ltx2_bounded_activation_offload` | Experimentally offload only evolving LTX checkpoint-boundary activations through bounded asynchronous pinned-memory transfers. Requires ordinary gradient checkpointing and is disabled by default. |
| `--ltx2_activation_offload_max_inflight N` | Maximum queued layer calls before CPU scheduling waits for CUDA work (default 2). |
| `--ltx2_activation_offload_keep_trailing N` | Keep the final N block-boundary activations resident because backward consumes them first (default 2). |
| `--ltx2_activation_offload_min_mb MB` | Offload only boundary activations at least this large (default 1 MiB). |
| `--blockwise_checkpointing` | Checkpoint transformer blocks individually and reload block state around backward. Lowest peak VRAM, but heavy CPU↔GPU traffic and recompute. |
| `--blocks_to_checkpoint N` | Number of final transformer blocks selected for blockwise checkpointing. With `--ltx2_partial_gradient_checkpointing` and no offload/swap, only these final N blocks use ordinary activation checkpointing. |
| `--offload_optimizer_during_validation` | Offload CUDA optimizer state to CPU during validation and sample previews (off by default) |
| `--ffn_chunk_target` | `none`, `all`, `video`, or `audio` — enable FFN chunking for selected modules (`none` selects no target, distinct from `--ffn_chunk_size 0`) |
| `--ffn_chunk_size N` | Chunk size for FFN chunking (0 = disabled). This can help inference-specific memory cases, but is not recommended for training because measured autograd peak memory and runtime increased. |
| `--split_attn_target` | `none`, `all`, `self`, `cross`, `text_cross`, `av_cross`, `video`, `audio` — split attention target modules |
| `--split_attn_mode` | `batch` or `query` — split by batch dimension or query length |
| `--split_attn_chunk_size N` | Chunk size for query-based split attention (0 = default 1024) |
| `--ddp_find_unused_parameters` | Enable DDP unused-parameter detection for branchy LoRA targets (off by default) |
| `--gemma_bnb_use_local_rank` | For Gemma 8-bit/4-bit loading, pin the quantized model to this process's `LOCAL_RANK` GPU (off by default) |
| `--ltx2_fused_norm_rope` | Fuse Q/K RMSNorm and RoPE for eligible BF16 CUDA tensors. Typically saves about 6% step time; use when a CUDA extension can be built. Unsupported shapes and `torch.compile` fall back automatically. |
| `--ltx2_fused_norm_rope_backward` | Fuse the matching RoPE/RMSNorm backward. Usually adds another 1.5-3%; use together with `--ltx2_fused_norm_rope`. |
| `--ltx2_compact_av_cross_adaln` | Project repeated AV cross-AdaLN timesteps once and gather the results. Expect up to about 2% in AV training; useful when many tokens share a small set of conditioning timesteps. |
| `--ltx2_compact_av_cross_adaln_min_tokens N` | Minimum token count for compact AV cross-AdaLN (default 256). Raise it if small AV shapes regress. |
| `--ltx2_fp8_placement_scope` | Reuse verified module placement during one scaled-FP8 forward. Can save about 8% on short checkpointed workloads; use with `--fp8_base --fp8_scaled`. It has no effect without scaled FP8. |
| `--ltx2_attn_auto_dispatch` | Route large maskless `--sdpa` shapes to cuDNN-prioritized attention. Gains range from negligible on short sequences to about 27% on long sequences; masked and unsupported shapes fall back automatically. Backend BF16 rounding can differ. |
| `--ltx2_flash_attn_4` | Within `--ltx2_attn_auto_dispatch`, use optional FlashAttention-4 for eligible large maskless BF16/FP16 CUDA attention with head dimension 128. Short, masked, compiled, other-head-dimension, and unsupported calls retain the existing PyTorch/cuDNN fallbacks. On the tested H100 long-video LoRA workload this improved throughput by about 12-14%; short shapes regress and are deliberately excluded. Requires a mutually compatible `flash-attn-4`, CuTeDSL, Quack, CUDA, and PyTorch installation. |
| `--ltx2_prompt_kv_checkpoint` | Keep video prompt K/V across activation-checkpoint recomputation. Can save about 9% when prompts are long and video sequences are short; use with gradient checkpointing and resident weights. It is disabled with activation/weight offload and does not optimize audio prompt K/V. |
| `--ltx2_prompt_kv_checkpoint_max_mb MB` | Limit preserved prompt K/V payload (default 1024 MiB). Lower it to trade recomputation savings for memory. |
| `--ltx2_padded_prompt_trim` | Remove the right-padded prompt tail shared by every sample before transfer and attention. Can save up to about 8% when cached prompts are heavily padded; it does nothing for all-valid prompts. |
| `--ltx2_partial_gradient_checkpointing` | Checkpoint only the final `--blocks_to_checkpoint N` blocks and retain earlier activations. Can improve short-sequence throughput by about 28% at higher VRAM cost; use only without offload, swap, or model parallelism. Keep N near 48 for long sequences. |
| `--ltx2_compile_inner_blocks` | With `--compile`, compile block compute while leaving checkpoint wrappers eager. Measured 18-21% higher throughput on short LoRA and about 9% at 13,728 video tokens; use for resident-weight LoRA. Dense FFT was about 2% slower and does not benefit. |
| `--ltx2_block_swap_async_backward` | Overlap pinned block transfers with backward compute. Measured 7.9-8.0% faster for 512x512x33 full fine-tuning; requires block swap, pinned memory, and gradient checkpointing. |
| `--ltx2_block_swap_trainable_ring` | Coalesce trainable FFT block transfers in a preallocated ring. Requires pinned block swap, gradient checkpointing, and fused backward. |
| `--ltx2_validate_training_tensors` | Scan cached inputs, predictions, and targets for NaN/Inf every step. Disabled by default to avoid GPU synchronization; enable it when diagnosing unstable training or suspect caches. |
| `--sdpa` | Use PyTorch scaled dot-product attention. |
| `--cudnn_attn` | Outside `torch.compile`, call PyTorch SDPA with backend priority cuDNN → flash → efficient → math. If the installed PyTorch lacks priority support, warn and use plain SDPA. Under `torch.compile`, use plain SDPA because the priority context is not traceable. Masks are passed to SDPA; the backend PyTorch ultimately selects is build/shape/hardware-dependent. |
| `--flash_attn` | Use FlashAttention 2 (requires `flash-attn` built for your CUDA + PyTorch). Key-padding masks use vectorized per-row varlen packing; measured about 6% faster for a mixed-length batch-4 prompt workload. |
| `--flash3` | Use the separately installed FlashAttention 3 interface. A reducible key-padding mask uses its variable-length function when present; if that function is absent, or the mask is query-specific, the code warns once and falls back to PyTorch SDPA. Package and hardware requirements depend on the installed FlashAttention build. |

The LTX-2 performance switches are disabled by default.

### Bounded Activation Offload
<sub>[↑ contents](#table-of-contents)</sub>

`--ltx2_bounded_activation_offload` is an LTX-only alternative to the existing broad checkpoint CPU-offload path. It selects only the evolving video/audio `x` tensors saved at block boundaries. Shared prompt, mask, timestep, and positional tensors stay on their existing path because copying them once per block would add traffic without releasing their original storage.

The selected tensors are copied to pinned host memory without mutating their source. Dense storage is transferred as a flat byte span, and the original shape and stride are restored during reload. Forward scheduling is capped by `--ltx2_activation_offload_max_inflight`; backward starts the previous block's reload before its checkpoint recomputation; and `--ltx2_activation_offload_keep_trailing` avoids needless round trips for the boundaries consumed first.

This path requires `--gradient_checkpointing`. It cannot currently be combined with `--gradient_checkpointing_cpu_offload`, `--blockwise_checkpointing`, `--ltx2_partial_gradient_checkpointing`, `--compile`, model parallelism, or remote stages. Ordinary LTX block swap remains supported. The feature is experimental: measure peak VRAM, host RAM, and step time on the target workload rather than assuming it is better than ordinary checkpointing.

**H2D-only block swap is opt-in:** it removes the redundant Device→Host copy
for frozen-base LoRA/LoHa/LoKr training. It requires CUDA, active block swap,
and `--gradient_checkpointing`; it is not valid for full fine-tuning and is
incompatible with checkpoint/activation CPU-offload modes. Invalid combinations
are warned about or rejected. See the [Block Swap guide](./block_swap.md).

### Blockwise Checkpointing
<sub>[↑ contents](#table-of-contents)</sub>

`--blockwise_checkpointing` checkpoints transformer blocks individually and reloads block state during backward. This shifts more state through CPU memory and increases recomputation cost.

Peak memory and step time are dataset-, bucket-, system-, and configuration-dependent. Increasing `--blocks_to_swap` moves more block state through host memory and generally trades device residency for transfer/recompute work; benchmark the exact target configuration.
`--blocks_to_checkpoint -1` applies blockwise checkpointing to all transformer blocks.

### Aggressive VRAM Optimization (8-16GB GPUs)
<sub>[↑ contents](#table-of-contents)</sub>

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
  --blockwise_checkpointing ^
  --blocks_to_checkpoint -1 ^
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
- `--use_pinned_memory_for_block_swap` speeds up block-swap transfers, but it is optional: remove it if you hit a block-swap crash (see [Troubleshooting](#troubleshooting), especially on RTX 5090 / 50xx)

This configuration trades main RAM for VRAM. It requires a large amount of main RAM: measured peak for the command above is about 80 GB, and the same command fails on a system limited to 72 GB. Allow at least 96 GB of installed RAM. Do not use it on a system with limited main RAM; see [Low Main-RAM Training](#low-main-ram-training) instead.

### Low Main-RAM Training
<sub>[↑ contents](#table-of-contents)</sub>

Block swap and CPU activation offload reduce VRAM by moving transformer blocks and activations into main RAM. On a system with limited main RAM the trade must be reversed: keep the model resident on the GPU and leave main RAM nearly empty.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --nf4_base ^
  --quantize_device cuda ^
  --gradient_checkpointing ^
  --sdpa ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 ^
  --network_alpha 16 ^
  --output_dir output
```

Rules for this configuration:

- **Do not set `--blocks_to_swap`.** Every swapped block is held in main RAM. Any non-zero value raises main-RAM usage well beyond what a small system has.
- **Do not use `--gradient_checkpointing_cpu_offload` or `--blockwise_checkpointing`.** Both move activations and block state through main RAM.
- **Use `--quantize_device cuda`.** Weights are then quantized directly on the GPU and the checkpoint is never staged in main RAM. `--quantize_device cpu` loads and quantizes on the host instead and raises main-RAM usage.
- **Disable sampling during training**, or precache the sample prompts (see [Precached Sample Prompts](#precached-sample-prompts)). Sampling otherwise loads Gemma, which does not fit alongside training on a small system.

Measured peak main-RAM usage for the command above is about 7.5 GB, and it does not scale with LoRA rank. Allow at least 12 GB of installed RAM so the operating system and data loading have headroom. The same configuration requires about 19 GB of VRAM.

That VRAM figure applies to this configuration only, because it keeps every block on the GPU. Block swap makes the opposite trade: adding `--blocks_to_swap 12` to the same command lowers VRAM to about 11 GB but raises main-RAM usage to about 26 GB. Choose the configuration that matches whichever of the two is scarcer on your system. `--fp8_base` and larger ranks raise the VRAM figure, and NF4 is the smallest base-weight format available.

#### Reducing the Main-RAM Cost of Block Swap
<sub>[↑ contents](#table-of-contents)</sub>

The main-RAM cost of block swap above comes from the loading step, not from the swapped blocks themselves: the checkpoint is staged in main RAM in full before any block is placed. `--ltx2_low_ram_load` instead writes each weight straight to the device it will be used from, so the model is never held whole in main RAM.

```bash
accelerate launch ... ltx2_train_network.py ^
  --nf4_base ^
  --quantize_device cuda ^
  --gradient_checkpointing ^
  --blocks_to_swap 16 ^
  --ltx2_low_ram_load ^
  ...
```

With `--blocks_to_swap 16` measured peaks are about 12 GB main RAM and 11 GB VRAM, against about 27 GB main RAM and 10 GB VRAM without the option. It trades roughly 1 GB of VRAM for roughly 15 GB of main RAM, which makes block swap usable on systems that cannot hold the whole checkpoint.

Notes:

- Requires `--blocks_to_swap`. Without block swap the model already loads directly onto the GPU.
- Supported for `--nf4_base`, `--fp8_base --fp8_scaled`, and unquantized bases. The int8, INT4-ConvRot and NVFP4 loaders do not place tensors individually, so the option is rejected for them rather than silently having no effect.
- Rejected together with `--loftq_init`, AWQ calibration, `--blockwise_checkpointing` and `--gradient_checkpointing_cpu_offload`, which all move state back through main RAM.
- Works with `--block_swap_h2d_only`, whose offloaded blocks are chosen differently from classic block swap.
- The option changes only where weights are written during loading, not what is computed.

### NF4 Quantization
<sub>[↑ contents](#table-of-contents)</sub>

NF4 (4-bit NormalFloat) quantization uses a 16-value codebook optimized for normally-distributed weights (QLoRA paper). Weights are stored as packed uint8 with per-block absmax scaling. It cuts transformer weight memory by ~75% vs BF16 (the DiT weights are roughly ~38–42 GB in BF16 and ~19–21 GB in FP8, version-dependent, so NF4 brings them to ~10 GB). Peak training VRAM additionally depends on resolution, frame count, block swap, and optimizer.

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

Use it like the original checkpoint: point `--ltx2_checkpoint` at the NF4 file. `--nf4_base` is still required (enables the runtime dequantization patch). Keep the `*.loftq_r<N>.safetensors` sidecar in the same directory as the NF4 checkpoint, and pass the same `--network_dim <N>` used to generate it so `--loftq_init` can load it:

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
- Pre-quantized checkpoints use the same NF4 packing/dequantization path as runtime dynamic quantization for the same block size. Exact byte equality can still depend on device, PyTorch, and CUDA behavior.

**Notes:**
- `--nf4_base` and `--fp8_base` are mutually exclusive.
- `--loftq_init` requires `--nf4_base`.
- `--awq_calibration` is experimental. Adds a per-layer division during forward passes. In synthetic tests, reduces activation-weighted error by ~3-5%; effect on real training quality has not been validated.
- Compatible with `--blocks_to_swap`, `--gradient_checkpointing`, and other training options. NF4 reduces block swap transfer size (4-bit vs 16-bit per weight).
- Quantization targets transformer block weights only. Embedding layers, norms, and projection layers remain in full precision.

### NVFP4 (FP4 E2M1) Checkpoints
<sub>[↑ contents](#table-of-contents)</sub>

NVFP4 (NVIDIA FP4, E2M1) is a 4-bit weight format used by some pre-quantized LTX-2 checkpoints (e.g. Lightricks releases). Transformer block weights are stored as packed uint8 (two FP4 values per byte) with two-level scaling: a per-block `weight_scale` (float8_e4m3) and a per-tensor `weight_scale_2` (float32). Early blocks, VAE, vocoder, connectors, biases, and norms stay in BF16.

NVFP4 checkpoints are detected automatically from the safetensors metadata (or from the presence of `weight_scale_2` keys) — there is no separate flag. Point the DiT checkpoint at the NVFP4 file and it is loaded with on-the-fly dequantization to the compute dtype during each `F.linear`, then a LoRA is trained over the frozen base, the same way as NF4. Auto-detection is skipped when `--nf4_base` or `--fp8_scaled` is set.

Weights are dequantized to the compute dtype and run through a standard `F.linear` (not a native FP4 matmul), so NVFP4 works on any BF16-capable GPU and does not require FP4 tensor cores (Blackwell). The 4-bit format is a memory optimization, not an FP4 compute path.

(Credit: [phazei](https://github.com/phazei) — [PR #87](https://github.com/AkaneTendo25/musubi-tuner/pull/87))

### int8 Base (Optimum-Quanto)
<sub>[↑ contents](#table-of-contents)</sub>

`--int8_base` loads a checkpoint that was quantized to int8 with [Optimum-Quanto](https://github.com/huggingface/optimum-quanto) (`qint8` weight-only) and trains a LoRA over the frozen int8 base. The int8 Linear layers run int8×int8 matmuls (`torch._int_mm`); layers the checkpoint left in bf16 (and the LoRA) run in bf16.

```bat
python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\model_qint8.safetensors ^
  --int8_base ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  (other training options)
```

To quantize a standard (bf16/fp16) checkpoint on the fly instead of supplying a pre-quantized file, use `--int8_base_dynamic` (same flag set otherwise; quantization is streamed, so the full bf16 model is never held in memory).

Notes:
- Requires LoRA training (`--network_module`); the int8 base is frozen.
- `--int8_base` / `--int8_base_dynamic` are mutually exclusive with each other and with `--fp8_base`, `--fp8_scaled`, and `--nf4_base`.
- For `--int8_base` the checkpoint must be an Optimum-Quanto `qint8` export. The model config is read from the checkpoint metadata when present, otherwise inferred from the weights.
- **Triton is required for the fused int8 kernels and the lower-memory path, and is not installed by default.** Install it separately: `pip install triton` on Linux, or [`triton-windows`](https://github.com/woct0rdho/triton-windows) on Windows. Without Triton, PyTorch >= 2.1 uses `torch._int_mm` with a transposed int8 weight copy. On PyTorch < 2.1, the fallback path dequantizes the int8 weight for `F.linear`.
- With Triton, the int8 dequantization is fused into one pass. The activation quantization (forward) and gradient quantization (backward) are also fused into single Triton kernels by default; set `LTX2_INT8_FUSED_QUANT=0` to force the eager multi-pass path. This also applies to `--fp8_w8a8 --w8a8_mode int8`.

Performance is hardware- and shape-dependent. Benchmark the target configuration with the intended backend, precision, attention mode, sequence length, and optimizer settings.

### INT8 ConvRot Base
<sub>[↑ contents](#table-of-contents)</sub>

`--int8_convrot_dynamic` quantizes eligible LTX-2 transformer block Linear weights to INT8 ConvRot while loading a standard bf16/fp16 checkpoint. ConvRot rotates weights offline with a group-wise Hadamard matrix, then rotates activations online before the dynamic int8 matmul. The base is frozen, so this path is for LoRA training.

```bat
python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev.safetensors ^
  --int8_convrot_dynamic ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  --int8_convrot_quality_report output\int8-convrot-quality.json ^
  (other training options)
```

To skip the load-time conversion on later runs, pre-quantize once and train from the saved INT8 ConvRot checkpoint (a reusable Comfy-compatible base):

```bat
python ltx2_quantize_int8_convrot.py ^
  --input_model path\to\ltx-2.3-22b-dev.safetensors ^
  --output_model path\to\ltx-2.3-22b-dev-int8-convrot.safetensors

python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev-int8-convrot.safetensors ^
  --int8_convrot_base ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  (other training options)
```

Options:
- `--int8_convrot_base`: load a pre-quantized INT8 ConvRot checkpoint with Comfy-compatible metadata (`.weight`, `.weight_scale`, `.comfy_quant`).
- `--int8_convrot_dynamic`: quantize a standard checkpoint at load time.
- `--int8_convrot_groupsize`: group size or comma list; default `auto` tries `256,64,16` and uses the largest divisor of each weight input dimension.
- `--int8_convrot_no_mse_clip`: use plain absmax row scales instead of MSE-optimal per-row clipping.
- `--int8_convrot_quality_report`: write per-layer reconstruction metrics during dynamic quantization.
- `--convrot_policy`: apply a shared per-layer INT8/INT4 ConvRot storage/compute policy. `quantize=false` keeps a matching weight in floating point during dynamic quantization; `compute=dequantize` keeps compressed storage but uses a transient dense floating-point matmul.
- `ltx2_quantize_int8_convrot.py --quality_report PATH`: write the same metrics while pre-quantizing. If omitted, the converter writes `<output>.quality.json`; use `--no_quality_report` to skip it.
- The standalone converter writes its output tensor-by-tensor into one standard safetensors file, so it does not retain the full converted `state_dict` in host RAM. Pass `--resume` from the initial invocation to keep an incomplete file and journal that can continue after interruption; the source checkpoint, policy, and output-affecting options must remain unchanged.

Quality reports include per-layer cosine similarity, MSE, MAE, max absolute error, and SQNR for the reconstructed weight (`dequantize + inverse rotate`) versus the source weight. Inspect `summary.min_cosine`, `summary.weighted_sqnr_db`, and the worst layers by `mse` or `max_abs_error`.

Reference calibration for `ltx-2.3-22b-dev.safetensors` from the original bf16/fp32 source, using `--int8_convrot_groupsize auto` and MSE clipping (weight-reconstruction metrics, not generation quality):

| Targeted layers / params | Weighted SQNR | Weighted MSE | Min / mean cosine | Max abs error |
|---:|---:|---:|---:|---:|
| 1344 / 18.52B | 41.27 dB | 1.56e-7 | 0.999895 / 0.999963 | 0.0229 |

Notes:
- Prefer converting from the original bf16/fp16 checkpoint when available. Scaled FP8 checkpoints are supported as sources and are dequantized through their `.weight_scale` tensors before INT8 ConvRot quantization, but any error already introduced by FP8 remains part of the source.
- Requires LoRA training (`--network_module`); full-parameter fine-tuning of the frozen INT8 ConvRot base is not supported.
- Mutually exclusive with `--int8_base`, `--int8_base_dynamic`, `--fp8_base`, `--fp8_scaled`, and `--nf4_base`.
- Targets LTX-2 transformer block attention/FFN Linear weights and leaves norms, embeddings, patch/projection heads, AdaLN, gates, and other precision-sensitive tensors unquantized.
- The runtime uses the same int8 backend selector as the existing int8 path (`--w8a8_backend torch|triton|cutlass`). Pre-quantized INT8 ConvRot defaults to `torch._int_mm` when available; pass `--w8a8_backend triton` or `--w8a8_backend cutlass` to force a backend.
- The forward ConvRot activation rotation + rowwise quantization and the backward grad-input scale + inverse ConvRot epilogue are fused into single Triton kernels by default (group sizes up to 256); set `LTX2_INT8_FUSED_QUANT=0` for the eager path. The fused path is Triton-first; set `LTX2_INT8_CONVROT_CUDA_QUANT=1` to build and use the native CUDA ConvRot kernels instead (one-time compile). Both fall back to eager when Triton/CUDA are unavailable.

### W4A4G4 Training
<sub>[↑ contents](#table-of-contents)</sub>

W4A4G4 training is an experimental LoRA path over a frozen 4-bit base. Its frozen quantized residual, optional frozen low-rank stabilizer, and trainable LoRA are structurally inspired by the triple-branch decomposition in [FourTune (arXiv 2607.05711)](https://arxiv.org/abs/2607.05711). Local containers, ConvRot handling, backend dispatch, and fallbacks are repository-specific; this documentation does not claim reproduction or result equivalence to the paper. In W4A4G4 mode, weights, forward activations, and backward grad-output are quantized to 4 bits, but the actual matrix multiply may be a native extension or a floating-point fallback depending on the selected path.

`--w4a4g4_container` selects the numeric format ("container") of the frozen base:

- `int4` — packed signed-INT4 weights with ConvRot Hadamard rotation. `auto` resolves a plain bf16/fp16 checkpoint to this container. W4A4G4/W4A4G8 selects the local CUTLASS backend and therefore requires CUTLASS headers (`LTX2_CUTLASS_INCLUDE_DIR`) plus a working CUDA extension toolchain.
- `nvfp4` — packed NVFP4 E2M1 weights with no rotation. The default emulated provider dequantizes for floating-point matrix multiplication. The optional Triton `tl.dot_scaled` provider is gated to compute capability `>= 10.0`, requires explicit selection, and has not been validated on Blackwell hardware in this repository.

Both containers expose the three-branch model, but they differ in numeric format, scaling, rotation, backend dispatch, and default stabilizer rank. `--w4a4g4_container auto` (the default) picks the container from `--ltx2_checkpoint`: a converter-produced int4cr checkpoint → `int4`, a converter-produced NVFP4 checkpoint → `nvfp4`, a plain bf16/fp16 checkpoint → `int4`. An explicit `int4`/`nvfp4` that contradicts a detected pre-quantized checkpoint format is rejected.

Within the `int4` container, three peer mode flags select the activation/gradient precision:

- `--w4a4g4`: 4-bit activations and gradients. For the int4 container this selects `cutlass` and enables the two fusion gates. The selected CUTLASS backend must build successfully; individual fusion helpers can still leave ineligible modules on their unfused implementations.
- `--w4a8`: 4-bit weights with 8-bit activations (int4 container only). Weights stay packed INT4 while forward activations and backward grad-output are quantized to row-wise INT8; the default backend routing is used and no fusion is implied.
- `--w4a4g8`: 4-bit weights/activations with row-wise INT8 grad-output. It uses the same backend, fusion gates, and stabilizer defaults as `--w4a4g4`; int4 container only.

All three mode flags accept either a plain bf16/fp16 checkpoint (quantized during load) or a matching converter output (packed weights loaded directly). Metadata/keys select the loader. Converter and on-the-fly paths share quantization code, but a nonzero SVD stabilizer uses randomized low-rank decomposition; matching tensors require the same input, options, device behavior, and RNG state. The `nvfp4` container has no W4A8 or W4A4G8 mode; use `--w4a4g4 --w4a4g4_container nvfp4`.

> [!WARNING]
> INT4 ConvRot LoRA training is experimental and changes numerical execution relative to bf16/fp16. `--w4a8` uses packed INT4 base weights plus row-wise INT8 activations/grad-output; `--w4a4g8` changes activations to 4 bits; `--w4a4g4` changes both activations and grad-output to 4 bits. This repository does not guarantee a particular quality delta. Validate loss, gradients, checkpoint loading, and representative samples on the target system.

On-the-fly from a bf16 checkpoint:

```bat
python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev.safetensors ^
  --w4a4g4 ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  --int4_convrot_quality_report output\int4-convrot-quality.json ^
  (other training options)
```

Pre-quantize once, then reuse (auto-detected — same flag):

```bat
python ltx2_quantize_int4_convrot.py ^
  --input_model path\to\ltx-2.3-22b-dev.safetensors ^
  --output_model path\to\ltx-2.3-22b-dev-int4-convrot.safetensors

python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev-int4-convrot.safetensors ^
  --w4a4g4 ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  (other training options)
```

Options:
- `--w4a4g4`: experimental W4A4G4 LoRA mode. In the `int4` container it selects 4-bit activation/grad-output quantization, the `cutlass` backend, and both fusion gates; see the prerequisites and fallback limits above.
- `--w4a4g4_container`: `auto` (default), `int4`, or `nvfp4` — the numeric format of the frozen base (see above). `auto` detects it from the checkpoint (int4cr → `int4`, NVFP4 → `nvfp4`, plain bf16/fp16 → `int4`).
- `--w4a8`: 4-bit weights with 8-bit activations (int4 container only); default backend routing, no implied fusion.
- `--w4a4g8`: 4-bit weights/activations with 8-bit gradients (int4 container only); the `--w4a4g4` forward with the `--w4a8` backward grad path, and the same backend/fusion/stabilizer defaults as `--w4a4g4`.
- `--w4a4g4_stabilizer_rank`: rank of the frozen low-rank stabilizer split off each weight when `--w4a4g4`/`--w4a4g8` quantizes a bf16/fp16 checkpoint on the fly. The effective default is container-dependent: `int4` → `0` (the ConvRot rotation already tames outliers), `nvfp4` → `32` (no rotation, so a stabilizer is used). Pass an explicit value to override either; `0` disables it. Ignored for a pre-quantized checkpoint (it carries its own stabilizer) and for `--w4a8`.
- `--int4_convrot_groupsize`: group size or comma list; default `auto` tries `256,64,16`. Unlike the INT8 ConvRot path, W4A4 pads the rotated input dimension internally, so the weight input dimension does not need to be divisible by the group size.
- `--int4_convrot_no_mse_clip`: use plain absmax row scales instead of MSE-optimal per-row clipping.
- `--int4_convrot_quality_report`: write per-layer reconstruction metrics during dynamic quantization.
- `--int4_convrot_scale_refine_steps`: alternating least-squares refinement of each row scale after the clipping search. `0` (default) preserves the existing quantizer exactly; `2` enables the opt-in refined calibration. It applies only while quantizing a standard checkpoint or in the standalone converter.
- `--int4_convrot_group_scales SIZE`: enable per-group INT4 weight scales using `SIZE` as the maximum K-group width. `0` (default) preserves the row-scale checkpoint and runtime exactly. Each layer halves the explicitly requested power-of-two size when needed to divide its padded input width. The packed INT4 codes remain unchanged in layout, while optional ratio/size buffers map each K-group onto the existing per-row INT8 accumulation grid.
- `--int4_convrot_group_ratio_q8`: with group scales enabled, store each positive group ratio as an int16 Q8.8 value selected from the same signed-INT4 rounding interval. This halves ratio checkpoint/VRAM storage while preserving `round(code * ratio)` for every code from -8 through 7. Existing float32-ratio checkpoints remain supported.
- `--int4_convrot_compare_group_scales SIZES`: measure an explicit comma-separated list such as `0,128,64` in the INT4 quality report. Every listed candidate is evaluated for every quantized layer, but none is selected or applied. The report records reconstruction metrics plus float32/Q8 ratio bytes for manual comparison. Requires `--int4_convrot_quality_report`; the standalone converter exposes the same option.
- `--convrot_policy`: shared ordered per-layer policy for INT8 and INT4 ConvRot. In addition to `quantize` and `compute`, INT4 dynamic quantization and the standalone converter accept explicit `group_scales`, `group_ratio_q8`, and `scale_refine_steps` fields.
- `--int4_convrot_awq_calibration`: compute dataset-independent AWQ-style per-input-channel scales before dynamic INT4 ConvRot quantization. The scales are based on weight-column importance with uniform synthetic activation assumptions, so they are reusable across LoRA datasets for the same checkpoint/config.
- `--int4_convrot_awq_scales`: with `--int4_convrot_awq_calibration`, save reusable scales to this safetensors path; if omitted, a checkpoint-adjacent `*.int4_convrot_awq_scales.safetensors` path is used. Without calibration, load and apply an existing scales file before dynamic quantization.
- `--int4_convrot_awq_alpha`: scaling strength for INT4 ConvRot AWQ (`0` = no effect, `1` = full column-importance scaling; default `0.25`).
- `--int4_convrot_activation_calibration_report`: write activation-aware per-layer diagnostics from real training batches. The report compares the actual INT4 ConvRot Linear output with the same packed INT4 weights dequantized through `F.linear`, so it measures activation/backend error rather than full original-weight error.
- `--int4_convrot_activation_calibration_batches`: number of training batches to measure for the activation calibration report (default `1`).
- `--int4_convrot_activation_calibration_regex`: optional module-name regex for limiting measured layers.
- `--int4_convrot_activation_calibration_max_rows`: maximum activation rows sampled per module call (default `128`; `0` measures all rows).
- `--int4_convrot_activation_calibration_max_layers`: maximum number of matched INT4 ConvRot layers to measure (default `0`, all matched layers).
- `--int4_convrot_activation_calibration_only`: write the activation calibration report and exit before optimizer updates.
- `ltx2_quantize_int4_convrot.py --quality_report PATH`: write the same metrics while pre-quantizing. If omitted, the converter writes `<output>.quality.json`; use `--no_quality_report` to skip it.
- The standalone `int4cr` and `comfy_convrot_w4a4` converters stream directly into one standard safetensors file instead of retaining the full converted `state_dict`. Pass `--resume` from the initial invocation to keep an incomplete file and journal; completed quantized layers are checkpointed immediately. With a nonzero stabilizer rank, `--resume_seed` supplies the deterministic per-layer SVD seed used by resumable conversion.

Notes:
- Requires LoRA training (`--network_module`); full-parameter fine-tuning of the frozen INT4 ConvRot base is not supported.
- `--w4a4g4`, `--w4a8`, and `--w4a4g8` are mutually exclusive with each other and with `--int8_base`, `--int8_base_dynamic`, `--int8_convrot_base`, `--int8_convrot_dynamic`, `--fp8_base`, `--fp8_scaled`, and `--nf4_base`.
- The `LTX2_INT4_CONVROT_*` environment variables below remain expert overrides. When set explicitly they win over the value implied by `--w4a4g4`/`--w4a8`/`--w4a4g8`, and a one-line warning records the override.
- Targets LTX-2 transformer block attention/FFN Linear weights and leaves norms, embeddings, patch/projection heads, AdaLN, gates, and other precision-sensitive tensors unquantized.
- `LTX2_INT4_CONVROT_BACKEND=auto` uses the torch bridge when its requirements are met. Optional values are `torch`, `wmma`, `wmma_hybrid`, `cutlass`, `cutlass_int8`, and `scalar`; `cutlass` requires headers from `LTX2_CUTLASS_INCLUDE_DIR` and a successful extension build.
- `LTX2_INT4_CONVROT_ACT_BITS`: `8` selects W4A8 (the `--w4a8` default; forward activations and backward grad-output are row-wise INT8), `4` selects the fully 4-bit W4A4 activation path (the `--w4a4g4` default). Setting it explicitly overrides the mode flag (for example `LTX2_INT4_CONVROT_ACT_BITS=8` with `--w4a4g4` runs the W4A8 activation path).
- `LTX2_INT4_CONVROT_GRAD_BITS`: bit-width used to quantize the backward grad-output, decoupled from the forward activation bits. `4` selects a 4-bit gradient, `8` selects a row-wise INT8 gradient (the `--w4a4g8` default). When unset it follows the activation bits, so `--w4a4g4` uses a 4-bit gradient and `--w4a8` an 8-bit gradient. Set it explicitly to override the mode flag (for example `LTX2_INT4_CONVROT_GRAD_BITS=8` with `--w4a4g4` is the same as `--w4a4g8`).
- `LTX2_INT4_CONVROT_BACKEND`: tensor-core backend routing. `--w4a4g4` implies `cutlass`; `--w4a8` uses `auto`. Set explicitly to override (values: `auto`, `torch`, `wmma`, `wmma_hybrid`, `cutlass`, `cutlass_int8`, `scalar`).
- `LTX2_INT4_CONVROT_FUSE_CUDA` / `LTX2_INT4_CONVROT_FUSE_LORA`: the fused native-CUTLASS W4A4 activation/epilogue path and the fused triple-branch LoRA path. `--w4a4g4` implies both `1`; `--w4a8` implies neither (they are W4A4-only, so the fused LoRA path stays on the eager down/up path under W4A8 even if enabled). Set either explicitly to override.
- `LTX2_INT4_CONVROT_FUSE=1` requests fused Triton kernels for the W4A8 rotation/quantization and rescale/inverse-rotation helpers. It requires Triton and CUDA, is off by default, and applies only to W4A8. The fused and eager implementations can differ numerically; no tolerance or training-equivalence guarantee is made here.
- `LTX2_INT4_CONVROT_FUSED_GEMV=0` disables the default grouped-checkpoint inference kernel for activation matrices with at most 16 rows. When enabled (the default), Triton reads packed INT4 weights directly, applies group ratios in registers, performs the INT8 tensor-core dot, and writes the scaled output without a full INT8 weight tensor. It uses fast Triton activation quantization and is not bit-exact with the eager PyTorch activation quantizer; set this variable to `0` for eager numerical parity or when diagnosing small-M output differences.
- No Blackwell performance claim is made for the native INT4 kernels.
- Reconstruction reports measure weight or observed activation/backend error; they do not establish generation quality or convergence.
- Group-scaled checkpoints remain backward-compatible with this trainer's `int4cr` loader but require its group-ratio-aware runtime. They are intentionally rejected by `--export_format comfy_convrot_w4a4`. The CUDA runtime decodes eight packed INT4 codes per load, applies one group ratio, writes one INT8 word, and then uses the tensor-core integer matmul. It does not use the native packed W4A4 fused-LoRA kernel. If the optional CUDA extension cannot be built, an exact chunked PyTorch expansion remains available but is a compatibility path rather than the production-performance backend.
- Group scales are primarily a reconstruction-quality/storage trade-off. On 84 real LTX-2.3 22B layers sampled across blocks 0, 24, and 47, group size 128 reduced weighted deployed reconstruction MSE by 15.1% versus refined row scales and improved every sampled layer. Float32 ratio buffers add about 579 MB (6.2%) to the roughly 9.28 GB packed target-layer payload; `--int4_convrot_group_ratio_q8` halves that ratio overhead to about 289 MB. The Q8.8 encoder preserved all signed-INT4 mappings across the 144,703,488 ratios in the matched full group128 checkpoint, and native CUDA unpack plus fused GEMV were bit-exact with their float32-ratio outputs on a real 4096x4096 layer. Group size 64 reduced MSE by 24.4% but doubled the uncompressed ratio overhead. These weight-only measurements motivate 128 as the starting point but are not a generation-quality or convergence result. On H100, a warm end-to-end group128 kernel comparison against the exact ai-toolkit ConvRot backend measured this runtime 1.66x faster for `M=512,N=2048,K=2048`, 1.70x for `512,4096,4096`, and 1.22x for `128,16384,4096`. The direct packed small-M GEMV was 1.24x, 1.53x, 1.29x, and 1.40x faster at `M=1,4,8,16` with `N=K=4096`. These are isolated kernel measurements on one software/hardware stack, not whole-training throughput guarantees.
- Standalone generation accepts `--int4_convrot_base` for a converter-produced `int4cr` checkpoint and `--int4_convrot_dynamic` for on-load quantization of a standard checkpoint. The dynamic inference path also accepts `--int4_convrot_groupsize`, `--int4_convrot_no_mse_clip`, `--int4_convrot_scale_refine_steps`, `--int4_convrot_group_scales`, `--int4_convrot_group_ratio_q8`, and `--quantize_device`. Prepacked grouped checkpoints carry their group ratios and ratio dtype, so load them with `--int4_convrot_base` without repeating either group option.
- For diagnosis, set `LTX2_INT4_CONVROT_WEIGHT_ONLY=1` with `--w4a4g4`/`--w4a8`/`--w4a4g8` to dequantize the packed INT4 weights and run normal `F.linear` without INT4 activation quantization. This is not the fast W4A4 path; it separates packed-weight quality from activation-quantization quality when init samples look degraded.
- INT4 ConvRot AWQ scales multiply weight columns before packing and divide activations by the reciprocal factors at runtime. Before rounding this rescaling is algebraically cancelling; INT4 quantization still changes the layer output.
- AWQ scale files are tied to the base checkpoint, target layer set, group-size policy, and model mode. They are not tied to the LoRA dataset identity.
- For layer ranking, use both reports: `--int4_convrot_quality_report` estimates static packed-weight reconstruction error, while `--int4_convrot_activation_calibration_report` measures runtime activation/backend error on the observed training activation distribution.
- Activation calibration performs extra reference matmuls during the measured forward pass. Keep the batch count small, and use `--int4_convrot_activation_calibration_regex` or `--int4_convrot_activation_calibration_max_layers` when memory is tight.

**Per-layer ConvRot policy.** A policy is an ordered JSON rule set; later matching rules override earlier fields. Patterns match module names without the trailing `.weight`. Omitted INT4 fields inherit the corresponding CLI value; there is no automatic parameter selection:

```json
{
  "format": "ltx2_convrot_policy_v1",
  "defaults": {
    "quantize": true,
    "compute": "quantized",
    "group_scales": 128,
    "group_ratio_q8": true,
    "scale_refine_steps": 2
  },
  "rules": [
    {"pattern": "transformer_blocks.*.ff.net.0.proj", "group_scales": 64},
    {
      "pattern": "transformer_blocks.*.attn.to_out.0",
      "compute": "dequantize",
      "group_scales": 0,
      "group_ratio_q8": false
    },
    {"pattern": "transformer_blocks.0.*", "quantize": false}
  ]
}
```

`group_scales` must be `0` or a power of two of at least 16. `group_ratio_q8=true` requires the effective `group_scales` value to be nonzero. `scale_refine_steps` must be zero or greater. These three fields affect only conversion from a floating-point source; a prepacked checkpoint already carries its per-layer codes, scales, and ratio dtype. INT8 paths ignore the INT4-only fields. The converter records a mixed checkpoint through per-layer buffers rather than replacing explicit rules with a preset.

For a report-only matrix without changing the selected global or policy values:

```bat
python ltx2_quantize_int4_convrot.py ^
  --input_model path\to\ltx-2.3-22b-dev.safetensors ^
  --output_model path\to\ltx-2.3-22b-dev-int4-convrot.safetensors ^
  --convrot_policy path\to\int4-convrot-policy.json ^
  --int4_convrot_compare_group_scales 0,128,64 ^
  --quality_report output\int4-convrot-quality.json
```

The JSON section `group_scale_comparisons` contains `selection: "none"`, aggregate rows for each requested size, per-layer candidate measurements, both ratio-storage byte counts, and whether every measured ratio has an exact Q8.8 mapping. Candidate order is preserved. The separate `applied_parameters` array records the effective requested/resolved group size, ratio dtype choice, and refinement steps for each checkpoint layer, so the measured candidates cannot be confused with the values that were actually written.

Generate exact layer overrides from an existing INT8 or INT4 reconstruction report:

```bat
python ltx2_build_convrot_policy.py ^
  --quality_report output\int4-convrot-quality.json ^
  --output output\int4-convrot-policy.json ^
  --min_cosine 0.99 --min_sqnr_db 20 ^
  --action keep_bf16
```

Use `--action dequantize` to retain packed storage while removing activation/gradient low-bit matmul error on selected layers. Use `--action keep_bf16` before dynamic quantization (or pass the generated policy to a converter) to keep selected source weights in floating point. With an already pre-quantized checkpoint, `quantize=false` cannot restore the original weight; the loader retains compressed storage and applies `compute=dequantize`.

**Low-rank stabilizer branch.** A rank-`N` SVD component can be split off each weight before INT4 quantization and stored as two bfloat16 factors per layer (`.int4_stabilizer_l1`/`.int4_stabilizer_l2`). At runtime the factors are added back as a frozen high-precision branch in the forward pass and the gradient-input path, so the INT4 residual has a narrower value range while the low-rank component stays in bf16. Two ways to produce it: `ltx2_quantize_int4_convrot.py --stabilizer_rank N` bakes it into a reusable pre-quantized checkpoint (loaded automatically — no training flag), and `--w4a4g4`/`--w4a4g8` `--w4a4g4_stabilizer_rank N` computes the same factors on the on-the-fly (bf16 checkpoint) path. `--w4a8` and pre-quantized checkpoints ignore `--w4a4g4_stabilizer_rank` (the checkpoint carries its own stabilizer). The branch also applies to `LTX2_INT4_CONVROT_WEIGHT_ONLY=1` diagnostics; checkpoints without these tensors are unchanged. The tensors stay resident under `--blocks_to_swap` and add a small per-layer overhead proportional to the rank.

**LoRA EMA with INT4 ConvRot.** `--use_ema` enables an exponential moving average of the trainable LoRA adapter weights. This can make validation previews and saved LoRA checkpoints less sensitive to noisy individual optimizer steps, which is useful when training over a very low-precision frozen base such as INT4 ConvRot. It does not average or modify the packed INT4 base model itself.

When EMA is enabled, validation and training samples are rendered with the EMA adapter, then the live training weights are restored before the next optimizer step. Checkpoints save both the live LoRA (`name.safetensors`) and EMA LoRA (`name_ema.safetensors`) unless `--save_ema_only` is set.

Example starting point for short INT4 ConvRot LoRA trials:

```bat
--use_ema ^
--ema_decay 0.95 ^
--ema_update_after_step 100 ^
--ema_update_every 10 ^
--ema_cpu_offload
```

Use `--ema_cpu_offload` when VRAM is tight; for LoRA this stores only the adapter EMA shadow on CPU. If INT4 base-model-only samples differ strongly from bf16 base-model samples under identical sampling settings, EMA should be treated as a LoRA-stability tool rather than a base-quantization quality fix.

#### ComfyUI inference export (convrot_w4a4)

`ltx2_quantize_int4_convrot.py --export_format comfy_convrot_w4a4` writes the `convrot_w4a4` layout used by [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) instead of this trainer's int4cr prepack. It is a one-way inference export: it is **not** loadable by this trainer's W4 modes, and this trainer does not read it back.

The exporter implements a power-of-4 regular Hadamard rotation, symmetric per-row `absmax/7` INT4 quantization, signed-nibble packing, and JSON metadata without a runtime comfy-kitchen dependency. For each selected transformer-block Linear it writes `<base>.weight` (packed int8 storage), `<base>.weight_scale`, and `<base>.comfy_quant`. Consumer compatibility depends on a ComfyUI/comfy-kitchen version that supports this layout; no cross-version byte-equivalence claim is made.

```bat
python ltx2_quantize_int4_convrot.py ^
  --input_model path\to\ltx-2.3-22b-dev.safetensors ^
  --output_model path\to\ltx-2.3-22b-dev-convrot_w4a4.safetensors ^
  --export_format comfy_convrot_w4a4
```

Options and constraints:
- `--comfy_linear_dtype`: `int4` (default) or `int8` — the matrix-mult precision recorded per layer.
- Linear weights whose input dimension is not divisible by the ConvRot group size (256) are kept as 16-bit (unquantized) and reported in a summary line; everything outside the transformer blocks passes through unchanged.
- `--stabilizer_rank`, `--int4_convrot_awq_calibration`/`--int4_convrot_awq_scales`, and `--no_rotation` are rejected with this export: the `convrot_w4a4` format has no stabilizer/AWQ slot and always uses Hadamard rotation.

#### NVFP4 container

The `nvfp4` container (`--w4a4g4 --w4a4g4_container nvfp4`) stores selected frozen transformer-block Linear weights in packed NVFP4 E2M1 with no rotation. The frozen residual uses quantized inputs/grad-output, an optional bf16 low-rank stabilizer is added, and the trainable LoRA is the third branch. Only the LoRA adapter receives parameter gradients; the packed base and stabilizer are frozen.

The container works with either input, detected automatically from the checkpoint:

- a **bf16/fp16 checkpoint** — quantized to NVFP4 on the fly at load, one tensor at a time. `--w4a4g4_stabilizer_rank` sets the stabilizer rank (default `32` for this container; `0` disables it).
- a **converter output** from `src/musubi_tuner/ltx2_quantize_nvfp4.py` — the packed NVFP4 state is loaded directly. Converter and on-the-fly loading share quantization routines. With a nonzero stabilizer rank, matching output also requires the same input, options, device behavior, and RNG state used by the randomized low-rank decomposition.

On the fly (single command):

```bat
python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev.safetensors ^
  --w4a4g4 --w4a4g4_container nvfp4 --w4a4g4_stabilizer_rank 32 ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  (other training options)
```

Pre-quantize once, then reuse the converter output across runs (auto-detected — same flags):

```bat
python src/musubi_tuner/ltx2_quantize_nvfp4.py ^
  --input_model path\to\ltx-2.3-22b-dev.safetensors ^
  --output_model path\to\ltx-2.3-22b-dev-nvfp4t.safetensors ^
  --stabilizer_rank 32

python ltx2_train_network.py ^
  --ltx2_checkpoint path\to\ltx-2.3-22b-dev-nvfp4t.safetensors ^
  --w4a4g4 --w4a4g4_container nvfp4 ^
  --network_module networks.lora_ltx2 --network_dim 32 ^
  (other training options)
```

Notes:
- A converter output stores packed E2M1 nibbles, a per-`16x16`-tile fp8 scale, a per-tensor fp32 scale, `.nvfp4_shape` metadata, and optional stabilizer factors, and is loaded directly.
- `python src/musubi_tuner/ltx2_quantize_nvfp4.py --stabilizer_rank N`: split a rank-`N` randomized low-rank component off each weight before NVFP4 quantization and store it as two bfloat16 factors (`.nvfp4_stab_l1`/`.nvfp4_stab_l2`); the residual is quantized. `0` disables it. The factors load automatically when present. `--output_model` defaults to `<input>.nvfp4t.safetensors`; `--quality_report PATH` writes per-layer reconstruction metrics (defaults to `<output>.quality.json`; `--no_quality_report` skips it).
- The converter streams directly into one standard safetensors file. Pass `--resume` from the initial invocation to keep an incomplete file and journal; completed quantized layers are durable immediately, and `--resume_seed` makes their randomized stabilizer decomposition deterministic per layer.
- `LTX2_NVFP4_BACKWARD=bf16`: opt into a hybrid input-gradient path that reconstructs the frozen residual in bf16 and runs a bf16 GEMM. The default `quantized` mode preserves the W4A4G4 gradient path. The bf16 mode avoids transposing and repacking the packed weight for a scaled-MMA backward, but temporarily materializes the dense bf16 residual and changes gradient numerics; benchmark both step time and peak VRAM on the target system.
- Targets the same layer set as the INT4 ConvRot converter (transformer-block attention/FFN Linear weights); norms, embeddings, patch/projection heads, AdaLN, gates, and other precision-sensitive tensors stay in bf16.
- Weight scales are shared across each `16x16` tile so the transposed weight layout used in the backward pass reuses the same scale tiles without requantization. Activations and grad-outputs are quantized dynamically per NVFP4 `1x16` micro-block.
- `LTX2_NVFP4_BACKEND=auto` (default) selects `emulate` on every device. `emulate` round-trips activations/grad-output through the same quantizer, dequantizes the base, and performs a floating-point matrix multiply; it is a functional implementation, not a performance or bitwise substitute for scaled MMA. `LTX2_NVFP4_BACKEND=triton` explicitly selects the experimental `tl.dot_scaled` provider and rejects unsupported hardware/builds instead of silently selecting emulation.

### Model Version
<sub>[↑ contents](#table-of-contents)</sub>

- `--ltx_version 2.0|2.3`: Select target model version (default: `2.3`). Controls default behavior for version-dependent settings (e.g., `--shifted_logit_mode` defaults to `legacy` for 2.0, `stretched` for 2.3).
- `--ltx_version_check_mode off|warn|error`: How to handle mismatch between `--ltx_version` and checkpoint metadata (default: `warn`). The trainer reads checkpoint config keys (`cross_attention_adaln`, `caption_proj_before_connector`, `bwe` vocoder) to detect the actual version.

### Audio-Video Support
<sub>[↑ contents](#table-of-contents)</sub>

- `--ltx2_mode`, `--ltx_mode`: Training modality selector. Default is `v` (`video`). Values: `video`, `av`, `audio` (aliases: `v`, `va`, `a`).
- `--ltx2_audio_only_model`: Force loading a physically audio-only transformer variant (video modules omitted). Requires `--ltx2_mode audio`.
- `--separate_audio_buckets`: Keeps audio and non-audio items in separate batches (reduces VRAM for image/video-only batches).
- `--audio_bucket_strategy pad|truncate`: Audio duration bucketing strategy. `pad` (default) rounds to nearest bucket boundary and pads shorter clips with loss masking. `truncate` floors to bucket boundary and truncates all clips to bucket length (no padding or masking needed).
- `--audio_bucket_interval`: Audio bucket step size in seconds (default: `2.0`). Controls how finely audio clips are grouped by duration.
- `--min_audio_batches_per_accum`: Minimum number of audio-bearing microbatches per gradient accumulation window.
- `--audio_batch_probability`: Probability of selecting an audio-bearing batch when both audio and non-audio batches are available.
  - `--min_audio_batches_per_accum` and `--audio_batch_probability` are mutually exclusive.
  - In LTX-2 training these flags enable an audio-aware DataLoader sampler. They cannot be combined with `--accumulation_group_by`, because PyTorch only accepts one sampler for the DataLoader.
  - They can be combined with `--audio_loss_balance_mode`, per-modality caption dropout, `--audio_supervision_mode`, `--dcr`, and `--tarp`.
  - For a first mixed AV trial, `--audio_batch_probability 0.4` is a reasonable starting point. With gradient accumulation, use `--min_audio_batches_per_accum 1` instead.
- `--accumulation_group_by none|frames|bucket|dataset`: Opt-in ordering for gradient accumulation windows. `bucket` keeps each virtual batch on one full dataset bucket key (resolution/frame/audio), which is useful for mixed frame-length datasets with `--gradient_accumulation_steps > 1`.
- `--accumulation_group_remainder drop|pad|allow_mixed`: Handles buckets that do not divide evenly by the accumulation window. `drop` skips incomplete windows, `pad` repeats same-group batches, and `allow_mixed` keeps all batches but may mix groups in final windows.
- `caption_extension`: For directory datasets, use a different caption file suffix to select alternate captions. For example, `caption_extension = ".target.txt"` reads `clip.target.txt` for `clip.mp4`/`clip.png`. This is the directory-dataset equivalent of JSONL caption routing.
- `--caption_field`: For JSONL datasets, use this metadata field instead of `caption` when caching/training text embeddings. For I2V/reference datasets, store fields such as `target_caption` and `reference_caption`, then cache with `--caption_field target_caption` when the text conditioning should describe the target video/motion.
- `--caption_dropout_rate`: Probability of dropping ALL text conditioning (video + audio) for a sample during training (default: `0.0`, disabled). When triggered, the sample's text embeddings are zeroed and the attention mask is disabled except for one placeholder token kept active for shape/runtime safety. This trains the model to generate without text guidance and enables classifier-free guidance (CFG) at inference.
- `--video_caption_dropout_rate`: Probability of dropping only the video text conditioning while keeping audio text conditioning (default: `0.0`). AV mode only. Applied independently per sample before `--caption_dropout_rate`.
- `--audio_caption_dropout_rate`: Probability of dropping only the audio text conditioning while keeping video text conditioning (default: `0.0`). AV mode only. Applied independently per sample before `--caption_dropout_rate`.

The three dropout rates are independent. For a given sample, the per-modality dropout is applied first (on the separate `video_prompt_embeds` / `audio_prompt_embeds`), then the joint dropout is applied on the concatenated result. A sample can end up with: both modalities present, video-only, audio-only, or fully unconditional.

For balanced AV or `av_ic` runs, per-modality dropout can be combined with the audio-aware sampler, `--audio_loss_balance_mode ema_mag`, and `--dcr`. A modest first pass is `--video_caption_dropout_rate 0.05 --audio_caption_dropout_rate 0.10`.

### Loss Function Type
<sub>[↑ contents](#table-of-contents)</sub>

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

### Loss Weighting
<sub>[↑ contents](#table-of-contents)</sub>

- `--video_loss_weight`: Weight for video loss (default: 1.0).
- `--audio_loss_weight`: Weight for audio loss in AV mode (default: 1.0).
- Dataset config `video_loss_weight` / `audio_loss_weight` for a dataset: with the LoRA trainer they override the corresponding CLI weight; with the full fine-tune trainer they are multiplied by the CLI weight.
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

For audio-starved mixed datasets, `inv_freq` can be combined with `--audio_batch_probability` or `--min_audio_batches_per_accum`, plus `--audio_supervision_mode warn --audio_supervision_min_ratio 0.5`.

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

For balanced AV or `av_ic` runs, `--audio_loss_balance_target_ratio 0.33` can be combined with the audio-aware sampler, per-modality caption dropout, and `--dcr`. If audio improves while video or identity quality drifts, try a lower target such as `0.25` together with lower `--audio_lr`, lower `--audio_dim`, or modality freezing.

**`uncertainty` mode** — learnable homoscedastic uncertainty weighting ([Kendall et al., CVPR 2018](https://arxiv.org/abs/1705.07115)). Two log-variance scalars (`log_var_video`, `log_var_audio`) are added to the optimizer and learned jointly with LoRA weights. The combined loss is:

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

### Additional Audio Training Flags
<sub>[↑ contents](#table-of-contents)</sub>

- `--independent_audio_timestep`: Sample a separate timestep for audio (AV/audio modes only).
- `--audio_silence_regularizer`: When AV batches are missing audio latents, use synthetic silence latents instead of skipping the audio branch.
- `--audio_silence_regularizer_weight`: Loss multiplier for synthetic-silence fallback batches.
- `--audio_supervision_mode off|warn|error`: AV audio-supervision monitor mode.
- `--audio_supervision_warmup_steps`: Expected AV batches before supervision checks.
- `--audio_supervision_check_interval`: Run supervision checks every N expected AV batches.
- `--audio_supervision_min_ratio`: Minimum supervised/expected ratio required by the monitor.

For mixed datasets where `loss_a` is absent for long stretches, the supervision monitor can be combined with the audio-aware sampler and `--audio_loss_balance_mode inv_freq`. Use `warn` first to confirm the ratio before switching to `error`.

### Modality Freezing (G2D)
<sub>[↑ contents](#table-of-contents)</sub>

Adaptive modality freezing based on per-modality loss EMA ratio, inspired by G2D Sequential Modality Prioritization ([arXiv 2506.21514](https://arxiv.org/abs/2506.21514)). When one modality's loss is significantly lower than the other, its LoRA parameters are frozen (`requires_grad=False`) so the under-performing modality can train without gradient interference.

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

For audio-overfitting cases, modality freezing can be combined with `--audio_loss_balance_mode ema_mag`, a lower `--audio_lr`, and lower audio LoRA rank.

### Frozen Auxiliary LoRA During Training
<sub>[↑ contents](#table-of-contents)</sub>

Use frozen auxiliary adapters when a new LoRA should train while another LoRA remains active in the forward pass. This is useful for training an adapter on top of an existing behavior without permanently merging that behavior into the base model.

- `--frozen_network_weights <path> [path ...]`: Attach one or more LoRA weight files as frozen adapters during training.
- `--frozen_network_multiplier <float> [float ...]`: Optional multiplier per frozen adapter. Missing values default to `1.0`.

The frozen adapters are set to eval mode, are not added to the optimizer, and are not saved into the output LoRA. They only affect the training forward pass. Use `--network_weights` when you want to warm-start the LoRA that is being trained, and use `--base_weights` when you want to merge weights into the base model before training.

When narrowing a warm-start LoRA to a smaller target preset, add `--network_freeze_surplus_modules` to keep compatible extra modules from `--network_weights` active as frozen adapters. Only modules outside the active trainable network are frozen; trainable modules still load and save normally.

Example:
```bash
accelerate launch ... ltx2_train_network.py ^
  --network_module networks.lora_ltx2 ^
  --frozen_network_weights output/style_anchor.safetensors ^
  --frozen_network_multiplier 0.5 ^
  --network_dim 32 ^
  --output_name new_subject_lora
```

### Optimizers
<sub>[↑ contents](#table-of-contents)</sub>

LTX-2 training accepts optimizer selection through `--optimizer_type`; optimizer arguments are passed with `--optimizer_args "key=value" ...`. Optional package rows require that package to be installed in the active Python environment.

| Optimizer | Use when | Extra package | Fused backward | Notes |
| --- | --- | --- | --- | --- |
| `AdamW` | You want the standard PyTorch optimizer path. | No | No | Pass constructor options through `--optimizer_args`; on CUDA, `fused=True` is opt-in and can improve LoRA step time by about 4% while using standard fused AdamW arithmetic. |
| `AdamW8bit`, `PagedAdamW8bit`, `PagedAdam8bit` | You want bitsandbytes optimizer-state memory savings. | `bitsandbytes` | No | The paged variants require a bitsandbytes build that provides those classes. |
| `Adafactor` | You want Adafactor's factored optimizer state and scheduler behavior. | No | Yes | If `relative_step` is omitted, it defaults to `True`; relative-step mode uses the Adafactor scheduler path. |
| `CAME`, `CAMESimple`, `came_simple` | You want CAME without 8-bit optimizer-state quantization. | No | Yes | Supports `stochastic_rounding`, `use_cautious`, and related CAME args through `--optimizer_args`. |
| `CAME8bit`, `came_8bit` | You want CAME with block-wise 8-bit optimizer state for eligible tensors. | No | Yes | Supports the same CAME args plus 8-bit state settings such as `min_8bit_size` and `quant_block_size`. |
| `SinkSGD`, `SinkSGD_adv`, `sinksgd`, `sink_sgd`, `sinksgdadv` | You want Sinkhorn-normalized SGD with momentum for LoRA-style training. | No | Yes | Supports `momentum`, `nesterov`, `nesterov_coef`, `normed_momentum`, `weight_decay`, `sinkhorn_iterations`, `orthogonal_sinkhorn`, `orthogonal_gradient`, `spectral_normalization`, and `state_precision`. State precision modes are `auto`, `fp32`, and `bf16_sr`. Optional LR scaling requires explicit args such as `scale_lr_with_grad_accum=True` or `scale_lr_with_effective_batch=True`. (Credit: [Ada123-a](https://github.com/Ada123-a) — [PR #80](https://github.com/AkaneTendo25/musubi-tuner/pull/80)) |
| `ProdigyPlusScheduleFree`, `ProdigyPlus`, `PPlus` | You want the Prodigy Plus schedule-free optimizer. | `prodigy-plus-schedule-free` | Yes | Opt-in only. The trainer fills missing Prodigy Plus constructor args with the recommended defaults below, but does not change `--learning_rate`, `--lr_scheduler`, or clipping unless you pass those values or use the dashboard preset. (Credit: [Ada123-a](https://github.com/Ada123-a) — [PR #80](https://github.com/AkaneTendo25/musubi-tuner/pull/80)) |

Recommended Prodigy Plus LoRA starting point:

```bash
--optimizer_type ProdigyPlusScheduleFree ^
--learning_rate 1.0 ^
--lr_scheduler constant ^
--max_grad_norm 0 ^
--optimizer_args betas=(0.9,0.99) beta3=None weight_decay=0.0 weight_decay_by_lr=True use_bias_correction=False d0=1e-6 d_coef=1.0 prodigy_steps=0 use_speed=False eps=1e-8 split_groups=True split_groups_mean=False factored=True factored_fp32=True use_stableadamw=True use_cautious=False use_grams=False use_adopt=False d_limiter=True stochastic_rounding=True use_schedulefree=True schedulefree_c=0.0 use_orthograd=False use_focus=False
```

Optional post-step weight perturbation:

- `--weight_noise_mode none|relative|absolute`: Default is `none`. `relative` scales the perturbation by each trainable tensor's RMS; `absolute` uses the scale directly.
- `--weight_noise_scale <float>`: Noise scale used when weight perturbation is enabled. The default is `0.01`, but it has no effect while the mode is `none`.

The perturbation is applied only after a synchronized optimizer step, uses the optimizer's trainable floating-point tensors, and leaves the default training path unchanged.

### Per-Module Learning Rates
<sub>[↑ contents](#table-of-contents)</sub>

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

Independent per-group schedules are also opt-in:

- `--lr_group_scheduler_args <pattern=scheduler=...,key=value> ...`: assign complete schedules to regex-matched optimizer groups.
- Supported schedulers are `constant`, `constant_with_warmup`, `linear`, `cosine`, `cosine_with_restarts`, and `polynomial`.
- Optional settings are `warmup_steps`, `stable_steps`, `decay_steps`, `min_lr_ratio`, `num_cycles`, and `power`.
- Rules are evaluated in command-line order and the first match wins. Groups not matched by a rule retain the global scheduler.

Enabling this option splits LTX LoRA parameters into stable `unet_video`, `unet_audio`, and `unet_cross_modal` optimizer groups even when their initial learning rates are equal. More specific groups created by `--lr_args` keep their regex-derived names.

```bash
--learning_rate 1e-4 ^
--lr_scheduler cosine ^
--lr_warmup_steps 200 ^
--lr_group_scheduler_args ^
  cross_modal=scheduler=cosine,warmup_steps=100,stable_steps=200,decay_steps=1200,min_lr_ratio=0.10,num_cycles=0.5 ^
  audio=scheduler=linear,warmup_steps=50,decay_steps=800,min_lr_ratio=0.05 ^
  video=scheduler=cosine,warmup_steps=300,min_lr_ratio=0.02,num_cycles=0.5
```

Each rule starts from that optimizer group's configured learning rate, including `--audio_lr` or `--lr_args` overrides. The full rule set and current schedule position are included in training state, so full-state resume continues the same curves. Schedule-free optimizers and remote-stage training do not support independent group schedules.

When audio learns faster than video, a lower `--audio_lr` such as `3e-5` with a `1e-4` base LR can be combined with `--audio_loss_balance_mode ema_mag` and lower audio LoRA rank.

### Per-Module LoRA Rank
<sub>[↑ contents](#table-of-contents)</sub>

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

For AV runs where audio overfits before video, `--audio_dim 8 --audio_alpha 8` can be combined with lower `--audio_lr`, `--audio_loss_balance_mode ema_mag`, and modality freezing.

### Adaptive LoRA Rank
<sub>[↑ contents](#table-of-contents)</sub>

Implemented only for standard LoRA (`networks.lora_ltx2` / `networks.lora`).
Related paper: [Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation](https://arxiv.org/abs/2603.21884).

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

### Per-Module LoRA Dropout
<sub>[↑ contents](#table-of-contents)</sub>

LTX LoRA supports three distinct dropout mechanisms:

- neuron dropout masks adapter activations;
- rank dropout masks rank components;
- module dropout skips the complete adapter update for a module.

All three keep their existing global value as the fallback and can be overridden independently for video-only, audio-only, and cross-modal modules.

- `--network_dropout <float>`: Base dropout for all LoRA modules.
- `--network_args "audio_dropout=<float>"`: Optional dropout override for audio-only modules.
- `--network_args "video_dropout=<float>"`: Optional dropout override for non-audio video modules.
- `--network_args "cross_modal_dropout=<float>"`: Optional dropout override for cross-modal modules (`audio_to_video`, `video_to_audio`, `av_ca_*`).
- `--video_rank_dropout`, `--audio_rank_dropout`, `--cross_modal_rank_dropout`: Optional rank-dropout overrides. Unset values fall back to `--network_args rank_dropout=...`.
- `--video_module_dropout`, `--audio_module_dropout`, `--cross_modal_module_dropout`: Optional whole-module dropout overrides. Unset values fall back to `--network_args module_dropout=...`.

Example:
```bash
--network_dropout 0.10 ^
--network_args "audio_dropout=0.15" "video_dropout=0.05" "cross_modal_dropout=0.20" "rank_dropout=0.05" ^
--video_rank_dropout 0.02 --audio_rank_dropout 0.10 --cross_modal_rank_dropout 0.05 ^
--video_module_dropout 0.00 --audio_module_dropout 0.05 --cross_modal_module_dropout 0.02
```

Result:
- Audio-only modules → dropout `0.15`
- Video-only modules → dropout `0.05`
- Cross-modal modules → dropout `0.20`
- If no per-modality override is provided, modules keep the global `--network_dropout`

Precedence is: `cross_modal_dropout` override > `audio_dropout` override for audio-only modules > `video_dropout` override for video-only modules > global `--network_dropout`.

The same precedence applies independently to rank and module dropout. All dropout probabilities must be in `[0, 1)`. These controls affect training only and require the built-in LTX LoRA network.

### Per-Modality Optimizer Controls
<sub>[↑ contents](#table-of-contents)</sub>

Video, audio, and cross-modal adapter groups can use independent clipping norms and weight decay:

```bash
--max_grad_norm 1.0 ^
--video_max_grad_norm 1.0 --audio_max_grad_norm 0.5 --cross_modal_max_grad_norm 0.25 ^
--video_weight_decay 0.01 --audio_weight_decay 0.02 --cross_modal_weight_decay 0.00
```

- `--video_max_grad_norm`, `--audio_max_grad_norm`, `--cross_modal_max_grad_norm`: independently clip each adapter group. An unset group inherits `--max_grad_norm`; `0` disables clipping for that group.
- `--video_weight_decay`, `--audio_weight_decay`, `--cross_modal_weight_decay`: override the optimizer's weight decay for the selected adapter group. An unset group retains the optimizer default.

Providing any optimizer override splits adapter parameters into stable modality-tagged optimizer groups even when their learning rates are equal. The group tags coexist with independent LR schedules, and weight-decay values are stored in optimizer state for full-state resume. These controls are opt-in and require the built-in LTX LoRA network. Per-modality optimizer controls are not available with remote-stage training; per-modality clipping is also not available with model-parallel training.

### Standalone Inference
<sub>[↑ contents](#table-of-contents)</sub>

To generate with a trained LoRA outside of training, run `ltx2_generate_video.py` with the base checkpoint, Gemma, a prompt, and the LoRA weights:

```bash
python ltx2_generate_video.py ^
  --ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
  --ltx_version 2.3 ^
  --gemma_root /path/to/gemma ^
  --gemma_load_in_8bit ^
  --lora_weight output/ltx23_lora.safetensors ^
  --lora_multiplier 1.0 ^
  --prompt "a corgi running on the beach at sunset" ^
  --height 512 ^
  --width 768 ^
  --frame_rate 25 ^
  --sample_steps 30 ^
  --guidance_scale 3.0 ^
  --seed 42 ^
  --sdpa ^
  --fp8_base ^
  --fp8_scaled ^
  --output_dir output/inference
```

`--prompt` (or `--sample_prompts <file>`) and `--gemma_root` (or `--gemma_safetensors`) are required; omit `--lora_weight` to sample the base model. For LTX-2.0, pass the 2.0 checkpoint with `--ltx_version 2.0`.

`ltx2_generate_video.py` also accepts a few standalone-inference-only overrides that are not part of the training sample table:

- `--vae`: Use a separate VAE checkpoint for inference. If omitted, `--ltx2_checkpoint` is used for both DiT and VAE loading.
- `--vae_dtype`: Override the VAE runtime dtype for inference. If omitted, the script uses its default VAE dtype (`bfloat16`).
- `--reference_image`: Apply one global I2V reference image to all prompts in the current inference run.
- `--reference_video`: Apply one global V2V reference video to all prompts in the current inference run.
- `--reference_videos PATH [PATH ...]`: Apply ordered V2V references for multi-reference pure-V2V inference. It cannot be combined with `--reference_image` or `--reference_video`.
- If both `--reference_image` and `--reference_video` are supplied, `--reference_video` takes priority.
- Global `--reference_image` / `--reference_video` overrides replace conflicting per-prompt `image_path` / `v2v_ref_path` entries loaded from prompt files, and also clear any cached reference latents tied to those prompt entries before sampling.
- If the path passed to `--reference_image` has a video filename extension, the script treats it as a V2V reference and routes it through the video-reference path.
- Attention backend: `--sdpa` (default), `--flash_attn`, `--flash3`, `--xformers`, or `--sage_attn` (equivalently `--attn_mode {sdpa,flash,flash3,xformers,sageattn}`). `--sage_attn` runs the DiT attention through [SageAttention](https://github.com/thu-ml/SageAttention) (requires the `sageattention` package; masked attention falls back to SDPA). It is inference-only — training rejects it. (Credit: [phazei](https://github.com/phazei) — [PR #87](https://github.com/AkaneTendo25/musubi-tuner/pull/87))

- `--ltx2_causal_temporal_attention`: Use the same causal-by-video-time self-attention as a causal-trained adapter. It requires SDPA (`--sdpa` or `--attn_mode sdpa`) and a video generation mode. Keep it disabled for ordinary adapters. A matched long-form quality study found better persistent-marker stability but worse held-out event ordering and ordinary short-video CLIP/DINO consistency, so this remains an experimental continuation/AR-only option rather than a general training recommendation.

#### Rendering Trained Conditioning
<sub>[↑ contents](#table-of-contents)</sub>

To render a LoRA trained with a conditioning recipe, pass the matching inference inputs below. Each is a global override applied to every prompt in the run; conditioning paths left unset stay off. Audio-video inputs require `--ltx2_mode av`.

- **Reference / IC-LoRA**: `--ic_lora_strategy {auto,none,v2v,av_ic,video_ref_only_av,audio_ref_ic}` selects the trained reference strategy (`auto` infers it from `--lora_target_preset`). Supply the reference media with `--reference_video` (video reference) and/or `--reference_audio` (audio reference; requires `audio_ref_ic` or `av_ic`). `--av_cross_attention_mode {both,a2v_only,v2a_only,none}` selects the cross-attention direction inside `av_ic`.
- **Inpaint**: `--inpaint_mask <image|video>` with a source video (`--inpaint_source`, or `--reference_video`) keeps the masked region from the source and regenerates the rest. `--inpaint_invert` flips the mask polarity; `--inpaint_threshold` sets the binarization threshold.
- **Outpaint / spatial-crop**: `--outpaint_region x,y,w,h` keeps that pixel rectangle from `--outpaint_source` and generates the surrounding area.
- **Extend**: `--extend_prefix <video>` locks the prefix and generates the continuation; `--extend_prefix_frames` sets how many pixel frames are locked.
- **Directional**: `--drive_audio <audio>` (a2v) locks the full audio and generates video from it; `--drive_video <video>` (v2a) locks the full video and generates its audio. Pair with `--av_cross_attention_mode a2v_only` / `v2a_only`.
- **Audio extend / inpaint**: `--audio_prefix <audio>` (with `--audio_prefix_seconds`) locks an audio prefix and generates the rest; `--audio_inpaint_audio <audio>` with `--audio_inpaint_interval start,end` (seconds) keeps the audio outside the interval and regenerates inside it.

### Audio Quality Metrics
<sub>[↑ contents](#table-of-contents)</sub>

Enable with `--audio_metrics`. All logic is in `audio_metrics.py`. When disabled, the trainer does not run the audio-metrics code path.

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
| `sample_audio/clap_similarity` | Per-sample CLAP audio-text cosine similarity | transformers (already a dep) |
| `sample_audio/clap_similarity_mean` | Mean CLAP score over the complete sample manifest pass | transformers (already a dep) |
| `sample_audio/clap_similarity_ci95_low`, `sample_audio/clap_similarity_ci95_high` | Normal-approximation 95% confidence interval over manifest scores | transformers (already a dep) |
| `sample_audio/clap_reference_similarity` | CLAP audio-embedding cosine similarity between generated and held-out reference audio | transformers (already a dep) |
| `sample_audio/clap_reference_similarity_mean` | Mean generated/reference similarity over the complete manifest pass | transformers (already a dep) |
| `sample_audio/fad_clap` | Fréchet distance between generated and held-out reference CLAP distributions; lower is better | transformers (already a dep) |
| `sample_audio/av_onset_alignment` | Correlation between audio energy onsets and decoded-video motion | None |
| `sample_av/av_desync_seconds_mean` | Mean absolute AV offset predicted by Synchformer; lower is better | Synchformer checkpoint |
| `sample_av/av_sync_score_mean` | Mean `1 / (1 + desync_seconds)`; higher is better | Synchformer checkpoint |

CLAP model is lazy-loaded on the first sample and offloaded to CPU between uses.
Every manifest entry must specify an explicit seed when CLAP validation is
enabled. One `sample/av_validation_<step>.json` report is written per manifest
pass with aggregate metrics, runtime, prompt/seed metadata, and paths to the
generated audio/video artifacts. CLAP input is resampled to its required 48 kHz.
Generated/reference comparison is separately opt-in:

```bash
--audio_metrics --audio_metrics_args clap_reference_similarity=true
```

Every entry must then provide held-out audio through
`validation_reference_audio_path` in a JSON/TOML prompt manifest or
`--vra path/to/reference.wav` in a TXT prompt manifest. This field is used only
by validation metrics. Do not substitute `--ra`, which conditions generation
on the supplied audio and would invalidate a held-out comparison.

FAD-CLAP uses the same paired manifest:

```bash
--audio_metrics --audio_metrics_args clap_fad=true clap_fad_min_samples=513
```

The metric is emitted only when generated and reference embedding counts match,
the sample count is at least both `clap_fad_min_samples` and
`embedding_dimension + 1`, and both empirical covariance matrices have full
rank. Otherwise the JSON report records why the result is invalid and no
FAD-CLAP scalar is logged. The default minimum of 513 matches the default
512-dimensional CLAP backend. Treat FAD-CLAP as a comparative validation signal
and confirm its ranking against held-out human judgments before using it as a
quality gate.

Synchformer DeSync validation is separately opt-in:

```bash
--audio_metrics --audio_metrics_args \
  av_desync=true \
  av_desync_checkpoint=path/to/synchformer_state_dict.pth \
  av_desync_device=cpu
```

It scores the generated MP4 and WAV sidecar after the complete manifest pass.
`av_desync_device=cpu` is the VRAM-safe default; use `cuda` when the validation
environment has enough free VRAM. `av_desync_max_length_s` controls the scored
duration and defaults to 8 seconds. Every entry must be a video sample with an
explicit seed. The per-sample offsets, derived sync scores, failures, backend
configuration, and aggregate confidence intervals are retained in the same
validation JSON report.

### Timestep Sampling
<sub>[↑ contents](#table-of-contents)</sub>

See also the [timestep bucketing documentation](./advanced_config.md) for advanced timestep bucketing options.

- `--timestep_sampling shifted_logit_normal`: Recommended LTX-2 method — pass it explicitly (the argparse default is `sigma`; every example command above sets it). Uses a shifted logit-normal distribution where the shift is computed from latent sequence length. In normal video/AV training this means `latent_frames × latent_height × latent_width`; only `--ltx2_mode audio` uses the audio-only sequence-length path described below.
- `--timestep_sampling uniform`: Uniform sampling from [0, 1].
- `--logit_std`: Standard deviation for the logit-normal distribution (default: 1.0). Only used with `shifted_logit_normal`.
- `--min_timestep` / `--max_timestep`: Optional timestep range constraints. By default LTX-2 scales the sampled sigma into this range; with `--preserve_distribution_shape`, it rejection-samples from the original distribution and keeps only values inside the range.
- `--num_timestep_buckets`: Stratified timestep buckets are honored by LTX-2 `shifted_logit_normal` and `uniform` sampling.
- `--shifted_logit_mode legacy|stretched`: Sigma sampler variant (default: auto by `--ltx_version`; 2.0→`legacy`, 2.3→`stretched`).
  - `legacy`: `sigmoid(N(shift, std))`. Original behavior.
  - `stretched`: Normalizes samples between the 0.5th and 99.9th percentiles of the distribution, reflects values below `eps` for numerical stability, and replaces a fraction of samples with uniform draws to prevent distribution collapse at high token counts.
- `--shifted_logit_eps`: Reflection floor and uniform lower bound for `stretched` mode (default: `1e-3`).
- `--shifted_logit_uniform_prob`: Fraction of samples replaced with uniform `[eps, 1]` draws (default: `0.1`).
- `--shifted_logit_shift`: Override the auto-calculated shift value. Lower values (e.g., `0.0`) produce a symmetric distribution centered on medium noise (σ≈0.5) for learning fine details. Higher values (e.g., `2.0`) heavily right-skew the distribution toward high noise (σ≈0.9+) for learning global structure. If unset, it is computed dynamically from sequence length. By default, non-audio training uses the raw linear formula below (so short or long sequences can fall outside the anchor values), while `--ltx2_mode audio` clamps the auto-computed shift to the configured min/max shift bounds.
- `--shifted_logit_clamp_auto_shift`: Clamp non-audio auto-computed shifts instead of extrapolating outside the anchor range. This does not affect explicit `--shifted_logit_shift`.
- `--shifted_logit_min_shift` / `--shifted_logit_max_shift`: Clamp bounds for auto-computed shifts (defaults: `0.95` / `2.05`). Audio mode always applies these bounds to auto shifts; non-audio mode applies them only with `--shifted_logit_clamp_auto_shift`.

> [!NOTE]
> The `shifted_logit_normal` auto-shift uses a linear formula anchored at 0.95 for 1024 tokens and 2.05 for 4096 tokens, based on sequence length. By default, non-audio training extrapolates this formula outside those anchor points for shorter/longer sequences; for example, a single 768x768 image has latent sequence length `1 x (768/32) x (768/32) = 576`, which gives shift `0.7896`. In `--ltx2_mode audio`, the auto-computed shift is clamped to the configured min/max shift bounds, defaulting to `[0.95, 2.05]`.
> In `--ltx2_mode audio`, `shifted_logit_normal` still needs a sequence length to compute the shift, but there is no real video spatial dimension. Using full video resolution would inflate the sequence length and skew the shift upward. Instead, `--audio_only_sequence_resolution` (default `64`) provides a small fixed spatial footprint (4 tokens/frame), keeping the shift dominated by the temporal dimension (audio duration/FPS) which actually matters.
> In joint AV training (`--ltx2_mode av`), the auto shift still comes from the video latent geometry; the presence of audio latents does not change the shift calculation.

### LoRA Targets
<sub>[↑ contents](#table-of-contents)</sub>

Use `--lora_target_preset` to control which layers LoRA targets. For custom layer patterns and `--network_args` format, see the [LoRA documentation](./advanced_config.md#lora):

Preset scopes describe the modules they can match when those modules exist. In `--ltx2_mode v`/`video`, the trainer constructs a video-only transformer, so even the broad `t2v` preset cannot create audio or AV cross-modal adapters. Applying a video-mode LoRA to an AV checkpoint therefore leaves the checkpoint's audio layers untouched.

| Preset | Layers | Modality scope | Use Case |
|--------|--------|----------------|----------|
| `t2v` (default) | All attention (`to_q`, `to_k`, `to_v`, `to_out.0`) | Video + audio + cross-modal | Text-to-video default |
| `v2v` | All attention + video FFN + audio FFN | Video + audio + cross-modal | Video-to-video / IC-LoRA style |
| `lycoris` | Attention modules only | Depends on LyCORIS algorithm/config | LyCORIS adapter runs via `--network_module lycoris.kohya` |
| `video_sa` | Video self-attention (`attn1`) | Video only | Spatially-aligned controls (depth, pose, canny, inpaint) |
| `video_sa_ff` | Video self-attention + video FFN (`attn1`, `ff`) | Video only | Controls needing more capacity (local edit, cut-on-action) |
| `video_sa_ca_ff` | Video self-attention + cross-attention + video FFN (`attn1`, `attn2`, `ff`) | Video only | Text-guided controls (video detailing, camera-from-image, sparse tracks) |
| `audio` | Audio attn/FFN only | Audio only | Audio-only training (auto-selected when `--ltx2_mode audio`) |
| `audio_v2a` | Audio attn/FFN + `video_to_audio_attn` | Audio + V2A cross-modal | Audio preset plus `video_to_audio_attn` (audio queries over video tokens) |
| `audio_ref_ic` | Audio attn/FFN + bidirectional AV cross-modal | Audio + cross-modal | Audio-reference IC-LoRA |
| `av_ic` | All attention + video FFN + audio FFN (same as `v2v`) | Video + audio + cross-modal | Joint AV IC-LoRA. Use `--av_cross_attention_mode` for directional variants and `--av_multi_ref` when configuring a multi-reference AV IC run |
| `video_ref_only_av` | All attention + video FFN + audio FFN (same as `v2v`) | Video + audio + cross-modal | AV training with reference video only; target audio is generated on audio-bearing batches, and batches without cached audio train the video branch only |
| `full` | All linear layers for LoRA targeting | Video + audio + cross-modal | Highest capacity, larger file size |

**Modality scope matters when training on an AV checkpoint.** The `t2v`, `v2v`, `av_ic`, and `full` presets create LoRA weights for audio and cross-modal layers. If those layers receive no audio training signal (e.g., image/video-only dataset), the LoRA weights for audio modules are initialized but never meaningfully updated — applying such a LoRA can degrade the base model's audio capabilities. Use a `video_*` preset to restrict LoRA to video-branch modules only, leaving audio layers completely untouched. Connector layers (`Embeddings1DConnector`) are excluded by default; use `--train_connectors` to include them (see below).

The `audio` preset excludes `video_to_audio_attn`; `audio_v2a` includes it. Choose `audio_v2a` when the audio-side LoRA should also adapt how audio queries over video tokens. Choose `audio` when the run should leave `video_to_audio_attn` weights at base-model values — for example when later merging this LoRA with another LoRA that owns those layers.

To use custom layer patterns instead of a preset, use `--network_args`:
```bash
--network_args "include_patterns=['.*\.to_k$','.*\.to_q$','.*\.to_v$','.*\.to_out\.0$','.*\.ff\.net\.0\.proj$','.*\.ff\.net\.2$']"
```
Custom `include_patterns` override any preset.
When `include_patterns` is set (either explicitly or via a preset), only modules matching at least one pattern are targeted (strict whitelist behavior). Use `--lora_target_preset full` to target all linear layers.

### LoRA Target Estimation (`ltx2_estimate.py`)
<sub>[↑ contents](#table-of-contents)</sub>

`ltx2_estimate.py` runs the LTX forward/loss path on cached training batches and accumulates squared gradients ("Fisher-style importance") for LoRA-targetable weights.

- It uses `setup_parser_common()` plus `ltx2_setup_parser()`, so the normal LTX argument surface is available. In the estimator path, attention backend selection (`--sdpa`, `--cudnn_attn`, `--flash_attn`, `--flash3`, `--xformers`), `--blocks_to_swap`, `--gradient_checkpointing`, `--blockwise_checkpointing`, `--compile`, `--fp8_base` / `--fp8_scaled`, `--nf4_base`, and `--split_attn` are applied.
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

### Connector LoRA (`--train_connectors`)
<sub>[↑ contents](#table-of-contents)</sub>

Text connectors are stacked 1D transformer blocks (the layer count comes from the checkpoint config key `connector_num_layers`) between Gemma and the denoising transformer. They transform text embeddings before they reach the denoising model. `--train_connectors` includes these modules in LoRA training alongside the main transformer.

**Usage:** Cache with `--cache_before_connector`, train with `--train_connectors`. The same `--lora_target_preset` patterns apply to both transformer and connector layers. Connector and transformer LoRA weights are saved in one file. For standalone ComfyUI conversion, use `python -m musubi_tuner.ltx2_convert_lora_to_comfy`; connector weights in the converted file are detected by the supported loading path.

**Notes:** Keeps the frozen connector weights resident in VRAM (size scales with the checkpoint's connector layer/head config). Not compatible with LyCORIS. Connectors have `attn1` and `ff` only (no `attn2`).

## Regularization & Objectives
<sub>[↑ contents](#table-of-contents)</sub>

Optional losses and objectives layered on top of standard training. Each is opt-in and off by default.

### Preservation & Regularization
<sub>[↑ contents](#table-of-contents)</sub>

Optional techniques that constrain how the LoRA modifies the base model. All are disabled by default with zero overhead.

**Blank Prompt Preservation** — Prevents the LoRA from altering the model's blank-prompt output (used as the CFG baseline during inference):
```bash
--blank_preservation --blank_preservation_args multiplier=1.0
```

**Differential Output Preservation (DOP)** — Constrains the LoRA to reproduce the frozen model on trigger-free conditioning.

Fixed mode preserves one or more class prompts:
```bash
--dop --dop_args class=woman multiplier=1.0
--dop --dop_args "class=woman|person" "bank=woman outdoors;portrait of a person"
```
Use `|` for rotating class variants and `;` for additional exact prompts.

Caption-replacement mode preserves the scene and action described by every training caption while removing its trigger:
```bash
--dop --use_precached_preservation ^
  --dop_args mode=caption_replace trigger=sks class=woman loss=relative_huber
```
For a caption such as `sks woman walking through a rainy city`, the preservation condition is `woman walking through a rainy city`. Multi-concept mappings use `replace`:
```bash
--dop_args mode=caption_replace "replace=sks=>woman|person;sksdog=>dog"
```
Caption replacement requires a matching preservation cache. Fixed mode can encode live at startup or use the same cache path.

Important DOP values:

| Value | Default | Effect |
|-------|---------|--------|
| `mode` | `fixed` | `fixed` or per-caption `caption_replace` |
| `loss` | `mse` | `mse`, `relative_mse`, or spike-resistant `relative_huber` |
| `huber_delta`, `relative_eps` | `1.0`, `1e-6` | Robust-loss transition and denominator floor |
| `temporal_weight` | `0` | Preserves frame-to-frame output deltas |
| `inside_weight`, `outside_weight` | `1`, `1` | Weights DOP inside/outside the dataset video loss mask |
| `timestep_bins` | `0` | Cycles through deterministic sigma strata; zero reuses training timesteps |
| `timestep_min`, `timestep_max` | `0`, `1` | Sigma range used by timestep strata |
| `timestep_weight` | `none` | `none`, `snr`, `inverse_snr`, or `mid` |
| `adaptive_target` | `0` | Relative-drift target; zero keeps the fixed multiplier |
| `adaptive_ema`, `adaptive_rate` | `0.95`, `0.1` | Drift smoothing and multiplier response rate |
| `adaptive_min`, `adaptive_max` | `0`, `100` | Adaptive multiplier limits |
| `adaptive_warmup` | `0` | Steps before multiplier updates |
| `microbatch` | `0` | Preservation batch size; zero batches all conditions together |
| `anchor_size`, `anchor_interval` | `0`, `0` | Capture exact input/teacher-output tuples and replay one every N steps |
| `anchor_weight`, `anchor_path` | `1`, empty | Replay strength and optional persistent anchor-bank path |
| `strict_cache` | `true` | Reject prompt-rewrite or text-encoder identity mismatches |

Blank preservation and DOP share the same LoRA-OFF and LoRA-ON passes when both are enabled. Setting `microbatch` may split those passes to lower peak memory.
Training and validation log `dop/relative_drift`, `dop/preservation_score = 1 / (1 + drift)`, the temporal component, effective multiplier, and drift EMA. Compare validation task loss with preservation score when tuning the constraint instead of minimizing either value alone.

**Prior Divergence** — Encourages the LoRA to produce outputs that differ from the base model on training prompts, discouraging overly weak LoRA effects:
```bash
--prior_divergence --prior_divergence_args multiplier=0.1
```

**Audio DOP** — Preserves the base model's audio predictions on non-audio training steps. Requires `--ltx2_mode av`. On each non-audio batch, constructs silence audio latents, runs the transformer with LoRA OFF and ON, and minimizes MSE on the audio branch only. Zero cost on audio batches. Do not combine with `--audio_silence_regularizer`: it converts non-audio batches into audio batches, so audio DOP never fires (the trainer warns but does not error).
```bash
--audio_dop --audio_dop_args multiplier=0.5
```

**TARP (Temporally Aligned RoPE and Partitioning)** — Windowed cross-attention masks that restrict each video frame to temporally nearby audio tokens (A2V) and each audio token to its nearest video frame (V2A). Enforces temporal locality in the AV cross-attention without modifying model weights. Requires `--ltx2_mode av`. From [arXiv:2603.18600](https://arxiv.org/abs/2603.18600).
```bash
--tarp --tarp_args window_multiplier=3
```
`window_multiplier` controls the A2V window size: `s = multiplier * floor(audio_tokens / video_frames)`. Default 3 (each frame sees 3x its proportional share of audio). V2A always uses nearest-neighbour (s=1).

TARP can be combined with `--cts_lambda_video_driven` / `--cts_lambda_audio_driven` when lip sync or cross-modal alignment is the main target.

**DCR (Dynamic Context Routing)** — Per-sample gradient detachment in cross-attention for mixed audio/video batches. When a sample lacks audio (zero-padded) or uses a clean reference (sigma=0), DCR detaches that stream's cross-attention context, preventing gradient flow through absent or reference-only streams. Forward values are unchanged; only the gradient path is masked. Requires `--ltx2_mode av`. From [arXiv:2603.18600](https://arxiv.org/abs/2603.18600).
```bash
--dcr --dcr_args reference_detach=true
```
`reference_detach` (default `true`) additionally detaches the reference stream when its timestep sigma is exactly 0.

DCR can be combined with the audio-aware sampler, `--audio_loss_balance_mode ema_mag`, and per-modality caption dropout. For `av_ic` runs, keep `reference_detach=true` unless you specifically want gradients through clean reference streams.

**AV Cross Grad Surgery** — Branch-aware gradient scaling for LTX-2 AV cross-modal K/V projections. Forward values are unchanged; the hook only scales the backward path through selected `audio_to_video_attn.to_k`, `audio_to_video_attn.to_v`, `video_to_audio_attn.to_k`, and `video_to_audio_attn.to_v` projections. This is opt-in and requires `--ltx2_mode av`.
```bash
--av_cross_grad_surgery
```
Bare `--av_cross_grad_surgery` uses the OmniNFT-inspired A2V schedule `a2v=0:0,1-10:0.1,40-47:0.3` with `projections=k,v`. Custom schedules use `--av_cross_grad_surgery_args`:
```bash
--av_cross_grad_surgery --av_cross_grad_surgery_args a2v=0:0,1-10:0.1,40-47:0.3 v2a=40-47:0.3 projections=k,v
```
Schedule entries are comma-separated `block:scale` or `start-end:scale` selectors. Scales must be in `[0, 1]`. Blocks not listed keep normal gradients.

**AV Attention Loss Weighting** — Uses detached A2V/V2A cross-attention concentration to upweight selected video and audio denoising loss tokens during the existing forward pass. This is opt-in and requires `--ltx2_mode av`.
```bash
--av_attention_loss_weighting --av_attention_loss_max 1.5 --av_attention_loss_warmup_steps 400
```
The multiplier warms from `1.0` to `--av_attention_loss_max`; tokens without captured attention keep normal loss weight.

**Cross-Task Synergy** — Auxiliary AV losses with one modality clean (timestep=0), intended to provide cross-modal alignment targets. Adds one extra forward pass per AV batch for each enabled direction (up to two when both are set). From [Harmony](https://arxiv.org/abs/2511.21579).
```bash
--cts_lambda_video_driven 0.3 --cts_lambda_audio_driven 0.1
```
CTS can be combined with TARP. Keep it off unless AV sync or cross-modal alignment is the target and the extra compute is acceptable.

**TREAD** - Training-time token routing for LTX token streams. This is opt-in. Enable it with `--tread`; optional settings use `--tread_args`:
```bash
--tread --tread_args selection_ratio=0.5 start_layer_idx=3 end_layer_idx=-4
--tread --tread_args target=audio selection_ratio=0.5 start_layer_idx=3 end_layer_idx=-4
```
- Bare `--tread` uses defaults.
- Default `target` is `video`. `target=audio` routes audio tokens, and `target=both` routes both video and audio tokens. Audio targets require an audio-enabled LTX mode.
- Default `selection_ratio` is `0.5`.
- The default block range is `3/-4` for LTX-2.3 and `2/-2` for LTX-2.0.
- `--tread_args` accepts `target`, `selection_ratio`, `start_layer_idx`, and `end_layer_idx`; aliases are `modality`, `ratio`, `start`, and `end`.
- TREAD is training-only. Video targets require a video-enabled path; audio-only training must use `target=audio`.

**Differential Guidance** - Prediction-relative scaling for the video/main training target. (Credit: [Ada123-a](https://github.com/Ada123-a) — [PR #80](https://github.com/AkaneTendo25/musubi-tuner/pull/80)) This is opt-in:
```bash
--differential_guidance
```
It applies this transform before the normal video/main prediction loss:
```text
target = pred + scale * (target - pred)
```
The default scale is `3.0` when enabled. Use `--differential_guidance_scale` to override it. Scale `1.0` is unchanged; values above `1.0` strengthen the target delta; values between `0.0` and `1.0` soften it. The prediction is detached while constructing the target, so under MSE a scale of `s` multiplies the current gradient by `s` while the displayed guided loss grows by `s²`.

The legacy flag and scale remain constant by default. Optional controls make the strength selective:

```bash
--differential_guidance ^
--differential_guidance_scale 2.0 ^
--differential_guidance_schedule cosine ^
--differential_guidance_start_scale 1.0 ^
--differential_guidance_warmup_steps 200 ^
--differential_guidance_hold_steps 600 ^
--differential_guidance_decay_steps 200 ^
--differential_guidance_end_scale 1.0 ^
--differential_guidance_timestep_mode mid ^
--differential_guidance_timestep_floor 1.0
```

- `schedule=linear|cosine` interpolates from `start_scale` to the main scale, optionally holds it, then decays to `end_scale`. `constant` preserves the original behavior.
- `timestep_mode=mid|snr|inverse_snr` concentrates the extra scale at middle, low-noise, or high-noise timesteps. `timestep_floor` is the scale used where the selected weighting is zero.
- `--differential_guidance_residual_clip N` clips individual residual elements at `N` times the per-sample RMS. Zero disables clipping.
- `--differential_guidance_normalize_residual` RMS-normalizes each sample before scaling.
- `--differential_guidance_adaptive_target_norm N` adjusts a bounded multiplier toward video gradient norm `N`.
- `--differential_guidance_adaptive_target_ratio N` instead targets a video/audio gradient-norm ratio during AV training. Set only one adaptive target. `adaptive_ema`, `adaptive_rate`, `adaptive_min`, and `adaptive_max` control the response.

Training logs the original and guided video losses, their ratio, scheduled and effective scales, residual RMS, clipped fraction, adaptive multiplier, and gradient feedback. The feature adds no model forwards, needs no cache, affects training only, and does not change inference. It cannot be used in audio-only mode or together with HFATO.

| Technique | Extra forwards/step | Extra backwards/step | Starting value / multiplier |
|-----------|-------------------|---------------------|----------------------|
| `--blank_preservation` | +2 | +1 | 0.5 - 1.0 |
| `--dop` | +2 | +1 | 0.5 - 1.0 |
| `--prior_divergence` | +1 | 0 | 0.05 - 0.1 |
| `--audio_dop` | +2 (non-audio steps only) | +1 (non-audio steps only) | 0.3 - 1.0 |
| `--tarp` | 0 | 0 | N/A (mask only) |
| `--dcr` | 0 | 0 | N/A (gradient routing) |
| `--av_cross_grad_surgery` | 0 | 0 | A2V K/V: 0:0, 1-10:0.1, 40-47:0.3 |
| `--cts_lambda_*` | +1 per enabled direction | 0 | 0.1 - 0.3 |
| `--av_attention_loss_weighting` | 0 | 0 | Max 1.5, warmup 400 steps |
| `--tread` / `--tread_args` | 0 | 0 | N/A (routing only) |
| `--differential_guidance` | 0 | 0 | Constant scale 3.0 by default; optional schedule, timestep shaping, residual control, and adaptive gradient feedback |

> [!CAUTION]
> Blank preservation and DOP together still use +2 forwards and +1 backward because their conditioning is batched. A positive DOP `anchor_interval` adds one LoRA-ON forward and backward on replay steps. Audio DOP costs apply only on non-audio steps. CTS adds one forward per enabled direction. TARP, DCR, AV Cross Grad Surgery, AV Attention Loss Weighting, TREAD, and Differential Guidance add no extra passes; they modify the existing forward/backward in-place.

### CREPA (Cross-frame Representation Alignment)
<sub>[↑ contents](#table-of-contents)</sub>

Encourages temporal consistency across video frames by aligning DiT hidden states across frames via a small projector MLP. Only the projector is trained; all other modules stay frozen. CREPA uses hooks to capture intermediate features from the existing forward pass (no extra forward passes). Two modes are available: `dino` (based on [arXiv 2506.09229](https://arxiv.org/abs/2506.09229), aligns to pre-cached DINOv2 features from neighboring frames) and `backbone` (inspired by [SimpleTuner LayerSync](https://github.com/bghira/SimpleTuner), aligns to a deeper block of the same transformer).

Enable with `--crepa`. All parameters are passed via `--crepa_args` as `key=value` pairs:

```bash
accelerate launch ... ltx2_train_network.py ^
  --crepa ^
  --crepa_args mode=backbone student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2 schedule=constant warmup_steps=0 normalize=true
```

Optional EMA cutoff can disable the CREPA loss once the alignment score is already high enough:

```bash
accelerate launch ... ltx2_train_network.py ^
  --crepa ^
  --crepa_args similarity_threshold=0.90 similarity_ema_decay=0.99 threshold_mode=permanent
```

#### CREPA CLI Flags
<sub>[↑ contents](#table-of-contents)</sub>

| Flag | Type | Description |
|------|------|-------------|
| `--crepa` | store_true | Enable CREPA regularization |
| `--crepa_args` | key=value list | Configuration parameters (see table below) |

#### CREPA Parameters (`--crepa_args`)
<sub>[↑ contents](#table-of-contents)</sub>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `backbone` | Teacher signal source: `backbone` (deeper DiT block) or `dino` (pre-cached DINOv2 features) |
| `dino_model` | `dinov2_vitb14` | DINOv2 variant for dino mode. Must match the model used during caching. Options: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` |
| `student_block_idx` | 16 | Transformer block whose hidden states are aligned (0 to block_count−1; e.g. 0–47 for the 48-block 22B checkpoint) |
| `teacher_block_idx` | 32 | Deeper block providing the teacher signal (backbone mode only, must be > `student_block_idx`) |
| `lambda_crepa` | 0.1 | Loss weight for CREPA term. Recommended range: 0.05–0.5 |
| `tau` | 1.0 | Temporal neighbor decay factor. Controls how much nearby frames contribute vs distant ones |
| `num_neighbors` | 2 | Number of neighboring frames on each side (K). Frame f aligns with frames f-K..f+K |
| `schedule` | `constant` | Lambda schedule over training: `constant`, `linear` (decay to 0), or `cosine` (cosine decay to 0) |
| `warmup_steps` | 0 | Steps before CREPA loss reaches full strength (linear ramp from 0) |
| `max_steps` | 0 | Total training steps for schedule computation. Auto-filled from `--max_train_steps` if not set |
| `normalize` | `true` | L2-normalize features before computing cosine similarity |
| `similarity_threshold` | off | Enables EMA cutoff when set to a value from 0 to 1. CREPA is disabled after the smoothed alignment score reaches this value |
| `similarity_ema_decay` | 0.99 | Smoothing factor for the cutoff score. Higher values react more slowly |
| `threshold_mode` | `permanent` | `permanent` keeps CREPA off after cutoff; `recoverable` can resume CREPA if the score drops again |
| `cutoff_step` | 0 | Optional global step where CREPA is disabled regardless of score. 0 keeps this disabled |

#### CREPA Checkpoint & Resume
<sub>[↑ contents](#table-of-contents)</sub>

- The projector weights (~33M params for backbone mode) are saved as `crepa_projector.safetensors` inside the Accelerate `*-state` folder, so they are written only when training-state saving (`--save_state`) is enabled.
- When resuming with `--resume <state_dir> --crepa`, projector weights are automatically loaded from `<state_dir>/crepa_projector.safetensors` if present.
- EMA cutoff state is saved as `crepa_state.safetensors` when training state saving is enabled, so a permanent cutoff remains permanent after resume.
- The projector is **not needed at inference**; it is only used during training.

#### CREPA Monitoring
<sub>[↑ contents](#table-of-contents)</sub>

CREPA adds `loss/crepa` to TensorBoard/WandB logs. With EMA cutoff enabled, it also logs `crepa/weight`, `crepa/cutoff`, `crepa/alignment_score`, `crepa/similarity_self`, and `crepa/alignment_score_ema`. A healthy CREPA loss should:
- Start negative (cosine similarity is being maximized)
- Gradually decrease (more negative = stronger cross-frame alignment)
- Stabilize after warmup

Use EMA cutoff when CREPA is helpful early but you do not want it to keep pushing already-aligned clips for the whole run. Start with a high threshold such as 0.90-0.95; leave it off if you want the existing CREPA behavior.

#### CREPA Compatibility
<sub>[↑ contents](#table-of-contents)</sub>

- Works with block swap (`--blocks_to_swap`) — hooks fire when each block executes regardless of CPU offloading.
- Works with all preservation techniques (blank preservation, DOP, prior divergence).
- Works with gradient checkpointing.
- Projector params are included in gradient clipping alongside LoRA params.

#### Caching DINOv2 Features (Dino Mode)
<sub>[↑ contents](#table-of-contents)</sub>

Dino mode requires pre-cached DINOv2 features. Run this **after latent caching** (cache paths are derived from latent cache files). DINOv2 is not loaded during training; only cached tensors are read, so the DINO model itself adds no training VRAM.

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
- `--dino_repo_path`: Local `facebookresearch/dinov2` clone containing `hubconf.py`. Uses `torch.hub` with `source="local"` and avoids a GitHub fetch.
- `--torch_hub_dir`: Torch hub cache directory. Use this when the DINOv2 repo/weights are already pre-populated in a local cache.
- `--skip_existing`: Skip items that already have cached features.
- `--atomic_cache_writes`: Opt-in safety mode. Writes each DINO cache through a temporary sibling file before replacing the final path.

Output: `*_ltx2_dino.safetensors` files alongside your latent caches, containing per-frame patch tokens `[T, N_patches, D]`. For `dinov2_vitb14` at 518px input: `N_patches=1369`, `D=768`, so each frame adds ~2MB (float16). Disk usage scales linearly with frame count.

**Precaching preservation prompts:** Fixed DOP and blank preservation can avoid loading Gemma during training:
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
Caption-replacement DOP caches every unique rewritten caption in a deduplicated shared prompt bank. The `.pt` file contains the strict index; embeddings are lazy-loaded from hashed safetensors shards in the adjacent `<cache-name>_dop_bank` directory, with a bounded in-memory LRU:
```bash
python ltx2_cache_text_encoder_outputs.py --dataset_config ... --ltx2_checkpoint ... --gemma_root ... ^
  --precache_preservation_prompts --dop ^
  --dop_args mode=caption_replace trigger=sks class=woman

python ltx2_train_network.py ... --dop --use_precached_preservation ^
  --dop_args mode=caption_replace trigger=sks class=woman loss=relative_huber
```
Prompt-related values (`mode`, `class`, `trigger`, `replace`, and `bank`) must match between caching and training. Loss, weighting, adaptive, microbatch, and replay values are training-only and do not affect the prompt-cache identity. Cache mismatch is an error by default. The cache file is saved to `<cache_directory>/ltx2_preservation_cache.pt`; `--preservation_prompts_cache <path>` overrides it in both commands. Prior divergence does not need precaching.

### Self-Flow (Self-Supervised Flow Matching)
<sub>[↑ contents](#table-of-contents)</sub>

**Self-Flow is representation learning integrated directly into flow training.** It extends the representation-alignment idea used by REPA, replacing the external representation model with an EMA teacher derived from the flow model itself. For each training example it samples two noise levels. The student sees a mixture in which different tokens use different noise levels, while the teacher sees the same example uniformly at the cleaner of the two levels. The student performs the ordinary flow-prediction task and simultaneously learns to match a deeper teacher representation from an earlier student block. This encourages it to use cleaner context tokens to understand noisier regions and learn stronger global relationships.

The representation loss uses cosine similarity. `teacher_mode=ema` is the default and canonical implementation. `partial_ema` is a lower-memory approximation that tracks only the selected teacher block. `teacher_mode=base` remains available for adapter training as a related frozen-base consistency variant: it disables the adapter for the teacher pass. The optional **temporal/motion extension** (frame-neighbor and motion-delta terms) is LTX-specific.

> [!CAUTION]
> The practical value of Self-Flow for low-scale LoRA or full-parameter post-training has not yet been established by controlled quality comparisons in this trainer. Treat it as an experimental research option and compare it against an otherwise matched run with Self-Flow disabled.

Enable with `--self_flow`. Supported in `--ltx2_mode video`, `--ltx2_mode av`, and `--ltx2_mode audio`. In AV mode the video alignment runs by default and audio alignment is enabled with `lambda_audio > 0`. Audio-only mode requires `lambda_audio > 0` and the video/temporal lambdas set to zero. All parameters are passed via `--self_flow_args` as `key=value` pairs:

```bash
# Canonical EMA teacher, token-level alignment only
accelerate launch ... ltx2_train_network.py ^
  --self_flow ^
  --self_flow_args student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.8 mask_ratio=0.1 teacher_momentum=0.9999 dual_timestep=true

# Canonical audio objective (run with --ltx2_mode audio)
accelerate launch ... ltx2_train_network.py ^
  --self_flow ^
  --self_flow_args student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0 lambda_audio=0.8 lambda_temporal=0 lambda_delta=0 audio_mask_ratio=0.5 teacher_momentum=0.9999

# With temporal consistency (hybrid = frame alignment + motion delta)
accelerate launch ... ltx2_train_network.py ^
  --self_flow ^
  --self_flow_args lambda_self_flow=0.1 temporal_mode=hybrid lambda_temporal=0.1 lambda_delta=0.05 num_neighbors=2 temporal_granularity=frame
```

#### CLI Flags
<sub>[↑ contents](#table-of-contents)</sub>

| Flag | Type | Description |
|------|------|-------------|
| `--self_flow` | store_true | Enable Self-Flow regularization |
| `--self_flow_args` | key=value list | Configuration parameters (see table below) |

#### Self-Flow Parameters (`--self_flow_args`)
<sub>[↑ contents](#table-of-contents)</sub>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher_mode` | `ema` | Teacher source: `ema` (canonical EMA over all trainable transformer parameters), `partial_ema` (lower-memory approximation over the selected teacher block), or `base` (adapter-only frozen-base consistency variant; **not canonical Self-Flow**) |
| `student_block_idx` | `16` | Student feature block index (0-based; overridden by `student_block_ratio` when set) |
| `teacher_block_idx` | `32` | Teacher feature block index (must be `> student_block_idx`; overridden by `teacher_block_ratio` when set) |
| `student_block_ratio` | `None` | Ratio-based student layer selection. Resolves to `floor(ratio * depth)`. Takes priority over `student_block_idx`. |
| `teacher_block_ratio` | `None` | Ratio-based teacher layer selection. Resolves to `ceil(ratio * depth)`. Takes priority over `teacher_block_idx`. |
| `student_block_stochastic_range` | `0` | Randomly vary the student capture block ±N blocks each step. `0` = fixed block. Adds regularization diversity; note a single projector is shared across all depth variants. |
| `lambda_self_flow` | `0.1` | Loss weight for the video token-level representation alignment term |
| `lambda_audio` | `0.0` | Loss weight for audio representation alignment. When `> 0`, captures audio hidden states from the same student/teacher blocks and aligns them via a separate audio projector MLP. Supported in `--ltx2_mode av` and `--ltx2_mode audio`. `0` = disabled. |
| `mask_ratio` | `0.10` | Token mask ratio for dual-timestep mixing. Valid range: `[0.0, 0.5]` |
| `image_mask_ratio` | `None` | Optional mask ratio used automatically for single-frame image batches. `None` inherits `mask_ratio`; the canonical value is `0.25`. |
| `audio_mask_ratio` | `None` | Independent audio-token mask ratio. `None` inherits `mask_ratio`; the canonical value is `0.5`. |
| `frame_level_mask` | `false` | When `true`, mask whole latent frames instead of individual tokens. More semantically coherent masking for video. |
| `mask_focus_loss` | `false` | When `true`, compute the representation loss only on masked (higher-noise) tokens. Default: loss over all tokens. |
| `loss_type` | `one_minus_cosine` | Representation objective. The canonical value is `one_minus_cosine`; `negative_cosine` has the same gradient but shifts the reported and capped scalar loss by a constant. |
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
| `temporal_schedule` | `constant` | Schedule applied to **all** Self-Flow lambdas (`lambda_self_flow`, `lambda_audio`, `lambda_temporal`, `lambda_delta`): `constant`, `linear`, `cosine`, or `polynomial` decay |
| `temporal_warmup_steps` | `0` | Steps to linearly ramp all lambdas up from zero to full weight |
| `temporal_max_steps` | `0` | Step at which a decaying schedule reaches `schedule_end_weight`. `0` disables decay. If omitted for a decaying schedule, it defaults to `--max_train_steps` |
| `schedule_end_weight` | `0.0` | Final multiplier for decaying schedules. Valid range: `[0.0, 1.0]` |
| `schedule_power` | `1.0` | Positive exponent for `temporal_schedule=polynomial` |
| `schedule_cutoff_step` | `0` | Hard-disable all Self-Flow terms at this global step. `0` disables the hard cutoff |
| `similarity_cutoff` | `None` | Disable Self-Flow when the EMA video cosine reaches this threshold. Valid range: `[-1.0, 1.0]` |
| `similarity_ema_decay` | `0.99` | EMA decay for the similarity cutoff signal. Valid range: `[0.0, 1.0)` |
| `similarity_cutoff_mode` | `permanent` | `permanent` latches the cutoff; `recoverable` continues diagnostic capture and resumes the loss if EMA similarity drops below the threshold |
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

#### Notes
<sub>[↑ contents](#table-of-contents)</sub>

- Supported modes: `--ltx2_mode video`, `--ltx2_mode av`, and `--ltx2_mode audio`. In AV mode, each minibatch applies only the alignment terms for modalities present in that batch.
- Image-like training is supported through single-frame samples in `--ltx2_mode video`; `image_mask_ratio` lets mixed image/video runs use a distinct image mask without changing video masking. Keep `temporal_mode=off` for the canonical objective.
- Canonical hyperparameters are `lambda_self_flow`/gamma `0.8`, `teacher_momentum=0.9999`, and mask ratios `0.25` (image), `0.10` (video), and `0.50` (audio). They are shown explicitly above rather than imposed on existing configurations.
- Cost: one extra teacher forward pass per active train step. `teacher_mode=base` reuses the existing model with the adapter disabled instead of keeping a teacher-weight copy.
- Teacher modes: `ema` is canonical Self-Flow; `partial_ema` is a memory-saving approximation; `base` gives a pretrained-vs-finetuned consistency objective.
- Feature capture: the LTX transformer returns the requested video and audio hidden states directly. Self-Flow no longer depends on forward hooks, including under checkpointing.
- Temporal extension (LTX-specific): when `temporal_mode != off`, Self-Flow reshapes hidden states into latent frames and adds frame-neighbor and/or frame-delta consistency losses on top of the base token alignment loss.
- Granularity: `temporal_granularity=frame` uses mean-pooled per-frame features (cheaper, coarser). `temporal_granularity=patch` keeps spatial tokens for stronger temporal matching.
- Local patch matching: when `temporal_granularity=patch` and `patch_spatial_radius > 0`, each student patch can align to the best teacher patch inside a local spatial window, which is more tolerant to small motion and camera drift than strict same-patch matching.
- Soft matching: `patch_match_mode=soft` replaces hard local best-match selection with softmax-weighted neighborhood matching for smoother gradients.
- Multi-step motion: `delta_num_steps > 1` extends the delta loss beyond adjacent frames using exponentially decayed step weights.
- Motion-aware weighting: `motion_weighting=teacher_delta` upweights temporally active teacher regions, focusing the temporal loss on moving content.
- Scheduling: warmup, constant/linear/cosine/polynomial decay, end weight, hard cutoff, and similarity-based cutoff apply to all Self-Flow lambdas uniformly.
- Audio: when `lambda_audio > 0`, AV and audio-only modes build a separate dual-timestep student audio view and a homogeneous cleaner teacher view from the same clean latent and noise. Audio-only `t` and `s` use the same configured training distribution.
- Validation: the primary validation loss uses the normal homogeneous noising path. `val_self_flow_loss`, when logged, is a separate diagnostic and is not added to `val_loss`.
- State files (Accelerate `*-state` folder): `self_flow_projector.safetensors` and `self_flow_teacher_ema.safetensors`. Projectors are also registered on the trainable network so distributed wrapping and gradient synchronization include them.
- Resume: both state files are required and loaded automatically. Missing or incompatible Self-Flow state raises an error instead of reinitializing part of the objective.
- Evaluation/export: sample previews and validation use the Self-Flow EMA copy. Every student checkpoint also gets a normal loadable `*_self_flow_ema.safetensors` companion (and the usual `.comfy.safetensors` conversion for adapters). Projectors remain resume-only auxiliary state and are not included in deployable full-transformer checkpoints.
- Invalid or unsupported combinations fail during setup. Missing captures, missing audio features, non-finite losses, and failed temporal reshapes fail at the affected step instead of silently disabling the objective.
- Logged metrics include `loss/self_flow`, cosine terms, scheduled lambdas, `self_flow/schedule_scale`, `self_flow/cutoff_active`, `self_flow/similarity_ema`, masking/timestep statistics, and the isolated `val_self_flow_loss` diagnostic.

### HFATO (High-Frequency Awareness Training Objective)
<sub>[↑ contents](#table-of-contents)</sub>

Adapted from [ViBe (arXiv 2603.23326)](https://arxiv.org/abs/2603.23326). Experimental for LTX-2 — evaluate on your own data before relying on it.

**HFATO** is a training objective designed for image-only fine-tuning of video models. Before adding noise, clean latents are spatially degraded via the ViBe Stage 2 downsample-upsample operator: 5D trilinear interpolation with `scale_factor=(1.0, 0.5, 0.5)` and `align_corners=False`, destroying high-frequency details. The model is then supervised to reconstruct the original clean latents (x₀-prediction loss instead of standard velocity loss). Can be combined with the Relay LoRA workflow below for two-stage image-only training.

Enable with `--hfato`. Parameters are passed via `--hfato_args` as `key=value` pairs. Incompatible with any active `--ic_lora_strategy` (`v2v`, `av_ic`, `audio_ref_ic`, `video_ref_only_av`); only `none`/`auto` is allowed.

```bash
accelerate launch ... ltx2_train_network.py ^
  --hfato ^
  --hfato_args scale_factor=0.5
```

#### CLI Flags
<sub>[↑ contents](#table-of-contents)</sub>

| Flag | Type | Description |
|------|------|-------------|
| `--hfato` | store_true | Enable HFATO loss |
| `--hfato_args` | key=value list | Configuration parameters (see table below) |

#### HFATO Parameters (`--hfato_args`)
<sub>[↑ contents](#table-of-contents)</sub>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale_factor` | `0.5` | Spatial downsample ratio. `0.5` = halve each spatial dimension. Lower values destroy more high-frequency info and force stronger reconstruction. `0.25` is more aggressive. |
| `interpolation` | `trilinear` | Interpolation mode for downsample-upsample. `trilinear` matches the authors' ViBe implementation. `nearest`, `bilinear`, and `bicubic` are non-original experimental variants. |
| `probability` | `1.0` | Per-step probability of applying HFATO. `1.0` = always, matching the authors' Stage 2 objective. Values `< 1.0` mix HFATO and standard flow matching steps and are non-original. |

#### Relay LoRA Workflow (Image-Only Training)
<sub>[↑ contents](#table-of-contents)</sub>

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

The resulting LoRA is standard — no inference pipeline changes. In the ViBe Relay LoRA design, Stage 1 is used only to train the Stage 2 base; inference loads the original base model plus the Stage 2 LoRA, so the low-resolution bridge is not part of the deployed model.

HFATO can also be used standalone (without relay) as a spatial detail objective for image or video training.

### Latent Temporal Objectives
<sub>[↑ contents](#table-of-contents)</sub>

Adds two optional training-only terms to the LTX-2 video loss. The saved LoRA/checkpoint is loaded normally at inference.

- `--latent_temporal_weighting`: computes clean-latent frame deltas `||z[t+1] - z[t]||` and maps them to per-frame multipliers for the denoising loss.
- `--latent_delta_loss`: computes `x0_pred = noisy - sigma * video_pred` or uses predicted velocity, then matches temporal derivatives to the clean latent target: `Delta pred ~= Delta target`.

Paper basis: `--latent_temporal_weighting` follows Latent Temporal Discrepancy ([arXiv 2601.20504](https://arxiv.org/abs/2601.20504)). `--latent_delta_loss` is an LTX-2-specific auxiliary objective in this trainer, not a direct reproduction of a paper method.

Usage:

```bash
# LoRA defaults
accelerate launch ... ltx2_train_network.py ^
  --latent_temporal_weighting ^
  --latent_temporal_weighting_args alpha=0.5 mode=log normalize=mean clip_min=0.5 clip_max=2.0 ^
  --latent_delta_loss ^
  --latent_delta_loss_args weight=0.03 order=1 target=x0 sigma_min=0.05 sigma_max=0.85
```

When both flags are off, the trainer does not attach latent-temporal context and the training loss path is unchanged.

Behavior:

- `order=1` matches first-order frame deltas; `order=2` matches second-order deltas; `order=1+2` uses both.
- `sigma_min` / `sigma_max` gate only the extra delta loss.
- Delta loss matches target deltas; it does not minimize motion magnitude.
- HFATO uses its own x0 reducer, so latent temporal weighting is not applied to HFATO's primary loss.
- Token/reference IC paths are skipped unless they expose 5D video predictions.
- Logged metrics: `latent_temporal_weight_mean`, `latent_temporal_weight_min`, `latent_temporal_weight_max`, `loss/latent_delta`, `loss/latent_accel`, `loss/latent_temporal_extra`.

#### Weighting Args
<sub>[↑ contents](#table-of-contents)</sub>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Motion-weight strength |
| `mode` | `log` | Motion score transform: `log` or `linear` |
| `normalize` | `mean` | Score normalization |
| `clip_min` | `0.5` | Lower clamp before final mean rescale |
| `clip_max` | `2.0` | Upper clamp before final mean rescale |

#### Delta Loss Args
<sub>[↑ contents](#table-of-contents)</sub>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | `0.03` | Extra loss multiplier |
| `order` | `1` | `1`, `2`, `1+2`, or `both` |
| `target` | `x0` | Match derivatives of `x0` or raw `velocity` |
| `sigma_min` | `0.0` | Minimum sigma where the extra loss is active |
| `sigma_max` | `1.0` | Maximum sigma where the extra loss is active |
| `second_order_weight` | `0.5` | Multiplier for `order=2` term |
| `loss_type` | `mse` | `mse`, `l1`, `huber`, or `smooth_l1` |
| `huber_delta` | `1.0` | Huber beta for `huber` / `smooth_l1` |

## Conditioning & Control
<sub>[↑ contents](#table-of-contents)</sub>

Conditioning is configured through one recipe file (`--ltx2_conditioning_config recipe.toml`) that sets each modality's `is_generated` flag and a list of conditions; every entry is lowered onto the underlying `--ltx2_*` / `--ic_lora_strategy` / `--ltx2_train_direction` flags, so a recipe trains identically to the equivalent flags. The training mode is not a separate setting; it follows from `is_generated` and the conditions listed. Conditions are intrinsic (hold part of a generated modality clean — first frame, spatial crop, mask, extension), reference (concatenate a reference clip, selecting an `--ic_lora_strategy`), or directional (freeze a modality with `is_generated = false`).

### Composable Conditioning Recipe
<sub>[↑ contents](#table-of-contents)</sub>

A conditioning recipe is one TOML file describing the whole training scheme. Per modality (`[video]` / `[audio]`), set whether it is **generated** (the model learns to produce it) or **frozen** (kept clean and used as conditioning — directional training), and list **conditions**: keep the first frame clean, keep a rectangle clean, use a mask, extend a clip, or attach a reference. Each condition sets the same `--ltx2_*` flags as the equivalent CLI flag, so the recipe and the flags are interchangeable. Pass the file with `--ltx2_conditioning_config recipe.toml`. The dashboard Conditioning tab builds this file: add and remove conditions in the UI and it writes the TOML.

An ordinary LoRA (style, subject, motion) uses no conditions — that is the baseline. Conditioning expresses control and editing tasks as additions to it:

| I want to… | block + condition | key params | per-item data in the dataset |
|---|---|---|---|
| Plain style / subject / motion LoRA | *(none — the baseline)* | — | — |
| Animate from a still image (I2V) | `[video]` `first_frame` | `probability` | — |
| Inpaint a region | `[video]` `inpaint` | `probability`, `threshold` | `loss_mask_directory` |
| Outpaint (fill the surround) | `[video]` `spatial_crop`, or `inpaint` with `invert = true` | `probability`, `invert` | crop: `spatial_crop_region`; mask: `loss_mask_directory` |
| Extend a clip forward / backward | `[video]` `extend` | `prefix`, `suffix` | — (auto from target) |
| Generate / continue audio | `[audio]` `extend` or `inpaint` | per type | inpaint: `audio_cond_mask_directory` |
| Video-to-video (reference) | `[video]` `reference` → `v2v` | `probability` (dropout) | `reference_directory` |
| Audio + video reference | `[video]` + `[audio]` `reference` → `av_ic` | — | `reference_directory`, `reference_audio_directory` |
| Foley (video → audio) | `[video]` `is_generated = false` → `v2a` | — | — |
| Audio-driven video (a2v) | `[audio]` `is_generated = false` → `a2v` | — | — |

> A recipe is the *complete* set: any conditioner not listed is off. `first_frame` is the only one otherwise on by default (probability 0.1), so a recipe that omits a `[video]` `first_frame` **turns first-frame conditioning off** — except under a reference strategy or directional training, where `first_frame` is left at its default instead (see [Reference Conditioning (IC-LoRA)](#reference-conditioning-ic-lora) below). Add a `first_frame` condition to keep or retune it. Without a recipe, `--ltx2_first_frame_conditioning_p` keeps its `0.1` default, so plain video training is text-to-video with first-frame (I2V) conditioning on 10% of samples; set it to `0`, or use a generate-only recipe, for pure text-to-video.

`--ltx2_conditioning_config <file.toml>` is an optional recipe that drives three things from one file: (a) it enables the intrinsic conditioners (the in-clip clean-region conditioners — `first_frame`, `spatial_crop`, `inpaint`, `extend` — that need no external media), lowering each entry onto (i.e. setting) the individual `--ltx2_*` flags above so they compose; (b) a `reference` condition selects an IC-LoRA reference strategy (lowered onto `--ic_lora_strategy`); and (c) `is_generated = false` selects directional training (lowered onto `--ltx2_train_direction`). For the intrinsic conditioners it sets the same values as the individual flags rather than replacing the training mode. Condition data (the spatial-crop region, the mask directory, the reference media) stays in the dataset config.

Conditions live under `[video]` and/or `[audio]` blocks, each a list of `[[video.conditions]]` / `[[audio.conditions]]` tables:

```toml
[video]

  [[video.conditions]]
  type = "first_frame"      # keep frame 0 clean (on by default at 0.1; list it to keep/retune)
  probability = 0.9

  [[video.conditions]]
  type = "spatial_crop"     # outpaint via a clean rectangle (region in the dataset config)
  probability = 0.3
  invert = true

  [[video.conditions]]
  type = "extend"           # forward/backward video extension (auto-derived from the target)
  prefix = 2
  probability = 1.0

[audio]

  [[audio.conditions]]
  type = "extend"           # audio extension -> --ltx2_audio_extend_*
  suffix = 8
```

I2V (keep frame 0 clean):

```toml
[video]

  [[video.conditions]]
  type = "first_frame"
  probability = 0.9
```

Inpaint (on-disk mask):

```toml
[video]

  [[video.conditions]]
  type = "inpaint"          # alias: mask
  probability = 0.5
  threshold = 0.5
```

The mask comes from the dataset's `loss_mask_directory`.

End-to-end (inpaint LoRA), three files working together:

- **Dataset** (`dataset.toml`): a `video_directory` plus a `loss_mask_directory` of stem-matched masks (white = keep clean).
- **Recipe** (`recipe.toml`): one `[[video.conditions]]` of `type = "inpaint"`, `probability = 0.5`.
- **Command**: `accelerate launch ltx2_train_network.py --dataset_config dataset.toml --ltx2_conditioning_config recipe.toml --ltx2_mode video --ltx2_checkpoint <ckpt> --network_module networks.lora_ltx2 ...`

The mask lives in the dataset; the recipe turns it into a clean conditioning region at probability 0.5. For a reference run (`av_ic`), put `type = "reference"` in both `[video]` and `[audio]`, set the reference directories in the dataset TOML, and run with `--ltx2_mode av`.

Notes:

- The recipe is authoritative: a condition declared both in the recipe and as an explicit CLI flag is governed by the recipe (the CLI flag is overridden, logged at WARNING).
- Conditions must live under `[video]`/`[audio]` blocks; a top-level `[[conditions]]` list is rejected with a migration hint.

A `reference` condition selects an IC-LoRA reference strategy instead of an intrinsic flag. Its presence in a block is a zero-data marker; the strategy is derived from the recipe structure and lowered onto `--ic_lora_strategy`:

| Recipe structure | derived `--ic_lora_strategy` |
|---|---|
| `[video]` reference, no `[audio]` block | `v2v` |
| `[video]` reference + `[audio]` reference | `av_ic` |
| `[video]` reference + an `[audio]` block with no audio reference | `video_ref_only_av` |
| `[audio]` reference, no video reference | `audio_ref_ic` |

```toml
[video]

  [[video.conditions]]
  type = "reference"        # selects an IC-LoRA strategy (here: v2v)
  probability = 0.1         # optional: reference dropout -> --ic_lora_ref_probability
```

An optional `probability` on the `[video]` reference sets `--ic_lora_ref_probability` (reference dropout); the audio reference takes no parameters. Intrinsic conditioners compose with a reference: the `[video]` intrinsics (`spatial_crop` / `inpaint` / `extend`) and `first_frame` apply to the target video tokens under any reference strategy, and the `[audio]` intrinsics (`extend` / `inpaint`) apply to a generated audio target under the audio-bearing strategies (`av_ic` / `video_ref_only_av` / `audio_ref_ic`). A reference recipe leaves the intrinsics at their default values rather than disabling them. The derived strategy requires the matching `--ltx2_mode` (`v2v` needs `video`; `av_ic` / `video_ref_only_av` need `av`; `audio_ref_ic` needs `av` or `audio`). The reference media itself comes from the dataset config, exactly as for the equivalent `--ic_lora_strategy` CLI run — the recipe only selects the strategy.

A reference recipe auto-selects the matching LoRA target preset; do **not** also pass `--lora_target_preset` with a reference recipe (an explicit preset suppresses the auto-selection). Set `--lora_target_preset` only when **not** using a recipe.

`av_ic` (`[video]` reference + `[audio]` reference):

```toml
[video]

  [[video.conditions]]
  type = "reference"

[audio]

  [[audio.conditions]]
  type = "reference"
```

Run with `--ltx2_mode av`. The reference video and audio are set in the dataset TOML (see [IC-LoRA / Video-to-Video Training](#ic-lora--video-to-video-training)).

`audio_ref_ic` (`[audio]` reference, no video reference):

```toml
[audio]

  [[audio.conditions]]
  type = "reference"
```

Run with `--ltx2_mode av` or `audio`. The reference audio is set in the dataset TOML (see [Audio-Reference IC-LoRA](#audio-reference-ic-lora)).

Each block also accepts `is_generated` (bool, default `true`). Setting exactly one modality's `is_generated = false` freezes it as clean conditioning and trains [directionally](#directional-training-a2v--v2a): `[audio]` `is_generated = false` → `a2v` (freeze audio, generate video); `[video]` `is_generated = false` → `v2a` (freeze video, generate audio/foley). This lowers onto `--ltx2_train_direction` and requires `--ltx2_mode av` (enforced downstream, with the same feature-conflict rules as the flag). Setting both to `false` is an error (nothing is generated), and a `reference` cannot be combined with directional training.

```toml
[audio]
is_generated = false   # freeze audio, generate video  ->  --ltx2_train_direction a2v
```

```toml
[video]
is_generated = false   # freeze video, generate audio (foley)  ->  --ltx2_train_direction v2a
```

`v2a` auto-disables first-frame conditioning (see [Directional Training](#directional-training-a2v--v2a)).

Each recipe condition lowers onto the matching CLI flags (so the recipe and the flags are two ways to set the same thing); condition data that is inherently per-item stays in the dataset config:

| Conditioning | block + `type` | keys | CLI flags it lowers to | per-item data |
|---|---|---|---|---|
| Keep first frame clean (I2V) | `[video]` `first_frame` | `probability` | `--ltx2_first_frame_conditioning_p` | — |
| Outpaint (clean rectangle) | `[video]` `spatial_crop` | `probability`, `invert` | `--ltx2_spatial_crop` `_p` `_invert` | `spatial_crop_region` |
| Inpaint (on-disk mask) | `[video]` `inpaint` (alias `mask`) | `probability`, `invert`, `threshold` | `--ltx2_inpaint_mask` `_p` `_invert` `_threshold` | `loss_mask_directory` |
| Video extend (fwd/back) | `[video]` `extend` | `prefix`, `suffix`, `probability`, `prefix_p`, `suffix_p` | `--ltx2_extend_prefix_frames` `_suffix_frames` `_p` `_prefix_p` `_suffix_p` | — (auto from target) |
| Audio extend (fwd/back) | `[audio]` `extend` | `prefix`, `suffix`, `probability`, `prefix_p`, `suffix_p` | `--ltx2_audio_extend_prefix_frames` `_suffix_frames` `_p` `_prefix_p` `_suffix_p` | — (auto from target) |
| Audio inpaint | `[audio]` `inpaint` (alias `mask`) | `probability`, `invert`, `threshold` | `--ltx2_audio_inpaint_mask` `_p` `_invert` `_threshold` | `audio_cond_mask_directory` |
| Reference IC-LoRA | `[video]` / `[audio]` `reference` | `probability` (video only) | `--ic_lora_strategy` (derived) `--ic_lora_ref_probability` | video: `reference_directory` / `reference_directories`; audio: `reference_audio_directory` / `reference_audio_directories` |

`probability` may be written `p`; `inpaint` may be written `mask`; setting both `probability` and `p` on one condition is an error. Audio inpaint reads its mask from `audio_cond_mask_directory` (not `loss_mask_directory`).

Per-sample loss controls the masked-loss denominator. With the batch-global denominator every in-mask element is weighted equally across the batch; with the per-sample denominator each batch element is weighted equally regardless of how much of it is masked in. Per-sample is enabled automatically whenever conditioning is active (spatial_crop, inpaint, video/audio extend, audio inpaint, a reference strategy, or directional training); first-frame conditioning at its default probability does not trigger it. A conditioning run then weights every sample equally even when their masked fractions differ; a plain run uses the batch-global denominator. Pass `--ltx2_per_sample_loss` to force it on otherwise. To override under conditioning, set the recipe top-level key `per_sample_loss = false` (force batch-global) or `per_sample_loss = true` (force per-sample); it is a top-level recipe key, not a `[[conditions]]` entry. In the dashboard it is the Conditioning tab's **Per-sample loss** control (Auto / On / Off), where Auto is the automatic behavior described above.

The effective conditioning of a run is recorded in the checkpoint metadata (LoRA and full fine-tune) under `ss_ltx2_*` keys (only when non-default), regardless of whether it came from the recipe or the CLI flags.

In the dashboard, **Conditioning** is its own tab (next to Training). It is a recipe builder: add or remove conditions under a **Video** block and an **Audio** block — the same types as above (`first_frame`, `spatial_crop`, `inpaint`, `extend`, `reference` for video; `extend`, `inpaint`, `reference` for audio) — each with its own probability and parameters. Per modality you choose **Generate** or **Freeze** (freezing one modality is directional a2v / v2a), and you set the **Per-sample loss** policy (Auto / On / Off). A live TOML preview shows the recipe as you build it. The builder writes a `conditioning_config.toml` in the project directory and passes it as `--ltx2_conditioning_config`, so you compose the recipe visually instead of editing TOML by hand. The Audio block only accepts conditions when `--ltx2_mode` is `av` or `audio`. While a recipe is active it governs the conditioning, so the LoRA target preset on the Training tab is disabled (an inline note explains why); `--ltx2_mode`, the reference directories, and the keyframe / video-anchor guidance still apply. An **external recipe file** field (advanced) lets you point at a hand-written `.toml` instead; the builder takes priority and the external file is used only when the builder is empty.

### Intrinsic Conditions
<sub>[↑ contents](#table-of-contents)</sub>
Intrinsic conditions hold selected tokens of a generated modality clean: the clean latent is kept in place, its timestep is set to 0, and those tokens are excluded from the loss. Each applies per sample at its configured probability, so a batch mixes conditioned and unconditioned samples. Video supports first-frame, spatial-crop, on-disk mask, and extension; audio supports extension and mask. All compose with the reference strategies.

#### Spatial-Crop Region Conditioning (outpaint)
<sub>[↑ contents](#table-of-contents)</sub>

Optional training-time conditioning that marks a rectangular spatial region of the video latents as clean conditioning (timestep 0, excluded from the loss), so the model learns to generate the surrounding content (outpaint). Off by default; enable with `--ltx2_spatial_crop`.

The region is set **per dataset** in the dataset config as `spatial_crop_region = [y1, x1, y2, x2]` in **pixel** coordinates; it is floor-divided by the VAE per-axis scale factors to the latent grid at training time.

| Flag | Default | Description |
|---|---|---|
| `--ltx2_spatial_crop` | off | Master enable. When unset, training is unchanged and nothing is emitted. |
| `--ltx2_spatial_crop_p` | `0.0` | Per-sample probability that the region is applied to a given sample (a batch mixes conditioned and unconditioned samples). |
| `--ltx2_spatial_crop_invert` | off | Condition the area **outside** the rectangle instead of inside. |

Example dataset entry:
```toml
[[datasets]]
video_directory = "..."
spatial_crop_region = [0, 0, 256, 512]   # y1, x1, y2, x2 in pixels
```

Caveats:

- **Video target required**: `--ltx2_spatial_crop` is rejected for `--ltx2_mode audio` (there are no video latents to crop).
- **Combinable with IC-LoRA**: composes with the reference strategies (`v2v`/`av_ic`/`video_ref_only_av`/`audio_ref_ic`); the clean-latent paste rides the target video tokens and the per-token mask is OR-merged into each reference branch's target conditioning mask.
- **Empty region**: a region that floor-divides to less than one latent cell is logged as a warning and conditions no tokens.
- **Training-only**: no inference behavior is added.

#### On-Disk Mask Conditioning (inpaint/outpaint)
<sub>[↑ contents](#table-of-contents)</sub>

Optional training-time conditioning that uses a per-item mask to mark part of the video latents as clean conditioning (timestep 0, excluded from the loss), so the model learns to fill the rest (inpaint) or generate the surround (outpaint). Off by default; enable with `--ltx2_inpaint_mask`.

The mask is supplied through the dataset's existing `loss_mask_directory` (stem-matched per item, bucket-cropped and resized to the latent grid like any loss mask). While `--ltx2_inpaint_mask` is on, that mask is **binarized** at the threshold and used as a **conditioning** mask rather than a loss weight. **Convention:** white / `1` = conditioned (kept clean); use `--ltx2_inpaint_mask_invert` if your mask instead marks the region to generate.

| Flag | Default | Description |
|---|---|---|
| `--ltx2_inpaint_mask` | off | Master enable. When unset, training is unchanged and nothing is emitted. |
| `--ltx2_inpaint_mask_p` | `0.0` | Per-sample probability the mask is applied as conditioning (a batch mixes conditioned and unconditioned samples). With `0.0` the feature never applies. |
| `--ltx2_inpaint_mask_invert` | off | Condition the **complement** of the mask (generate the masked region instead of keeping it clean). |
| `--ltx2_inpaint_mask_threshold` | `0.5` | Binarization threshold (strict `>`); mask values above it are conditioned. |

Example dataset entry (the mask is a loss-mask directory):
```toml
[[datasets]]
video_directory = "..."
loss_mask_directory = "/path/to/masks"   # stem-matched per clip; white = conditioned region
```

Caveats:

- **Shares the loss-mask channel**: while `--ltx2_inpaint_mask` is on, the `loss_mask_directory` mask is consumed as a conditioning mask, **not** as a loss weight — you cannot use a separate soft loss-weight mask and an inpaint mask in the same dataset at once.
- **`--ltx2_inpaint_mask_p 0.0`**: the feature never applies and the mask falls back to a standard loss weight; a warning is logged. Set `p > 0` to use it.
- **Video target required**: rejected for `--ltx2_mode audio`.
- **Combinable with IC-LoRA**: composes with the reference strategies (`v2v`/`av_ic`/`video_ref_only_av`/`audio_ref_ic`); the binarized mask is OR-merged into each reference branch's target conditioning mask and consumed as conditioning (not a loss weight) when live.
- **Training-only**: no inference behavior is added.

#### Video Extension (prefix/suffix)
<sub>[↑ contents](#table-of-contents)</sub>

Optional training-time conditioning that keeps the first N (prefix) and/or last N (suffix) **latent** frames clean (timestep 0, excluded from the loss), auto-derived from the target latents (no dataset, no cache), so the model learns to extend a clip forward (from a prefix) or backward (from a suffix). Off by default; enabled when a span is set.

| Flag | Default | Description |
|---|---|---|
| `--ltx2_extend_prefix_frames` | `0` | Leading latent frames kept clean (forward extension). 0 = off. |
| `--ltx2_extend_suffix_frames` | `0` | Trailing latent frames kept clean (backward extension). 0 = off. |
| `--ltx2_extend_p` | `1.0` | Per-sample probability of applying the extension when enabled. |

Caveats:

- **Latent frames, not pixels**: a span of N latent frames covers `1 + 8*(N-1)` pixel frames (the causal VAE encodes the first frame on its own, then groups of 8).
- **Video target required**: rejected for `--ltx2_mode audio`.
- **Combinable with IC-LoRA**: composes with the reference strategies (`v2v`/`av_ic`/`video_ref_only_av`/`audio_ref_ic`); the clean prefix/suffix frames ride the target video tokens and are OR-merged into each reference branch's conditioning/loss mask.
- **Training-only**: no inference behavior is added.

#### Audio Extension (prefix/suffix)
<sub>[↑ contents](#table-of-contents)</sub>

The audio analog of video extension: keeps the first N (prefix) and/or last N (suffix) **audio latent timesteps** clean (timestep 0, excluded from the loss), auto-derived from the target audio latents (no dataset, no cache), so the model learns to continue audio forward or backward. Works in both audio-video (`--ltx2_mode av`) and audio-only (`--ltx2_mode audio`) training. Off by default; enabled when a span is set.

| Flag | Default | Description |
|---|---|---|
| `--ltx2_audio_extend_prefix_frames` | `0` | Leading audio latent timesteps kept clean (forward extension). 0 = off. |
| `--ltx2_audio_extend_suffix_frames` | `0` | Trailing audio latent timesteps kept clean (backward extension). 0 = off. |
| `--ltx2_audio_extend_p` | `1.0` | Per-sample probability of applying the extension when enabled. |

Caveats:

- **Audio latent timesteps**: the spans count audio latent timesteps (one conditioning token each), not seconds or video frames.
- **Audio target required**: rejected for `--ltx2_mode video`.
- **Combinable with the audio-bearing IC-LoRA strategies**: composes with `av_ic` / `video_ref_only_av` / `audio_ref_ic`; rejected only with `v2v` (video-only, no audio target).
- **Training-only**: no inference behavior is added.

#### Audio Inpaint / Mask Conditioning
<sub>[↑ contents](#table-of-contents)</sub>

The audio analog of the video inpaint mask: a per-item audio mask marks which **audio latent timesteps** are kept clean (timestep 0, excluded from the loss), so the model learns to generate the complement (audio inpaint). The mask is supplied through a dataset `audio_cond_mask_directory` of per-item time intervals (seconds), cached to a per-timestep binary mask — a **separate channel from the audio loss mask** (which has the opposite convention). Works in both `--ltx2_mode av` and `--ltx2_mode audio`. Off by default.

| Flag | Default | Description |
|---|---|---|
| `--ltx2_audio_inpaint_mask` | off | Enable audio inpaint conditioning (requires `audio_cond_mask_directory` in the dataset). |
| `--ltx2_audio_inpaint_mask_p` | `0.0` | Per-sample probability of applying the mask when present. 0 = never. |
| `--ltx2_audio_inpaint_mask_invert` | off | Condition the COMPLEMENT of the mask (generate the masked region instead). |
| `--ltx2_audio_inpaint_mask_threshold` | `0.5` | Binarization threshold (strict `>`). |

The dataset option `audio_cond_mask_directory` points to per-item interval files (`.json` with a `cond_mask_intervals`/`audio_cond_mask_intervals`/`intervals` key, or a `.txt` of `start end` seconds per line), matched by filename stem. Mask convention: 1 = conditioned (kept clean). Re-cache audio latents after adding/changing the directory. Set it on the **audio dataset** entry (the audio dataset reads it; it is ignored on video/image dataset entries, like `audio_loss_mask` intervals).

Caveats:

- **Audio target required**: rejected for `--ltx2_mode video`.
- **Combinable with the audio-bearing IC-LoRA strategies**: composes with `av_ic` / `video_ref_only_av` / `audio_ref_ic`; rejected only with `v2v` (video-only, no audio target).
- **Separate from the loss mask**: the conditioning mask is its own directory/channel; it does not reuse `loss_mask_directory`.
- **Training-only**: no inference behavior is added.

### Reference Conditioning (IC-LoRA)
<sub>[↑ contents](#table-of-contents)</sub>
A reference condition concatenates a reference clip to the target along the token axis; the reference tokens carry timestep 0 and separate positions (IC-LoRA concatenation). The recipe sets `--ic_lora_strategy` from where the reference is placed: video only → `v2v`; video and audio → `av_ic`; a video reference with a generated audio target that carries no reference → `video_ref_only_av`; audio only → `audio_ref_ic`. These four values are realizations of the single reference condition, not separate training modes, and each stays a valid `--ic_lora_strategy` for direct CLI use. Reference dropout (`--ic_lora_ref_probability`) drops the reference batch-wide and applies to the video-reference strategies only.

#### IC-LoRA / Video-to-Video Training
<sub>[↑ contents](#table-of-contents)</sub>

IC-LoRA (In-Context LoRA) trains the model to generate video conditioned on a reference image or video.

Reference frames are encoded as clean latent tokens (timestep=0) and concatenated with noisy target tokens during training. The model attends across both sequences, using the reference as conditioning context. At inference, the same concatenation scheme is applied. Position embeddings are computed separately for reference and target, allowing different spatial resolutions via `--reference_downscale`.

##### Step 1: Prepare Dataset
<sub>[↑ contents](#table-of-contents)</sub>

Create a dataset with a matching reference/source directory. For both video and image IC-LoRA datasets, use `reference_directory`. Each reference file must share the same filename stem as its corresponding training sample:

```
videos/                    references/
  scene_001.mp4              scene_001.png     # reference for scene_001
  scene_002.mp4              scene_002.jpg
  scene_003.mp4              scene_003.mp4     # video references also work
```

References can be images (single frame) or videos (multiple frames).

##### Step 2: Dataset Config
<sub>[↑ contents](#table-of-contents)</sub>

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

In the dashboard, AV IC behavior is configured from the **Conditioning** tab: adding an audio-and-video reference condition selects the `av_ic` strategy automatically (the LoRA target preset on the Training tab is then disabled). Under **Reference fine-tuning (AV)** on that tab:
- AV cross-attention direction: `both` / `a2v_only` / `v2a_only` / `none`
- **Multi-Ref AV** when the dataset uses the plural reference directory fields above

##### Step 3: Cache Latents
<sub>[↑ contents](#table-of-contents)</sub>

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

`--reference_downscale 2` encodes references at half spatial resolution (e.g., 384px for 768px target). Position embeddings on the reference spatial axes are scaled by the factor so they map into the target coordinate space. When downscaling is enabled, LTX2 buckets are aligned to `32 * reference_downscale` pixels so cached reference dimensions remain exact `/32` latent-grid multiples instead of being rounded down.

**Temporally-subsampled references** (`--reference_temporal_scale`):
```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --reference_frames 193 ^
  --reference_temporal_scale 2 ^
  --device cuda
```

`--reference_temporal_scale 2` stores the reference at half its temporal resolution — it keeps the first frame, then every other frame — and rescales the reference's time positions so each reference frame aligns to a real target frame at train and inference time. Use it when the reference can be represented at a lower frame rate than the target, to reduce the reference token count.

The reference must subsample cleanly onto the target's temporal grid (the VAE packs 8 frames into one temporal latent, with the first frame standalone): `--reference_frames` must be `8*k + 1` (e.g. 9, 17, 25, 193) with the factor dividing `k`, and the reference must span the same number of frames as the target. Incompatible combinations are rejected at cache time with the required values. Pass the same `--reference_temporal_scale` during training so the factor is recorded in the checkpoint metadata. Not combinable with multiple references (the streams are concatenated along time).

##### Step 4: Cache Text Encoder Outputs
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

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

If you used `--reference_downscale` or `--reference_temporal_scale` during caching, also pass them during training:
```bash
  --reference_downscale 2
  --reference_temporal_scale 2
```

##### Step 6: Sample Prompts
<sub>[↑ contents](#table-of-contents)</sub>

Append `--v <path>` after the prompt text in your sampling prompts file to specify the V2V reference for each prompt. Both images and videos are supported:

```
A woman walking through a forest --v references/scene_001.png --n blurry, low quality
A cat sitting on a windowsill --v references/scene_002.mp4 --n distorted
```

The `--sample_include_reference` flag shows the reference side-by-side with the generated output in validation videos.

**Inpaint (composable with the `--v` reference).** Append `--im <mask_path>` to a prompt line to regenerate only a masked region while clean-pasting the rest from the `--v` source — the inference-time form of the training-side `inpaint` condition. The same LoRA handles both maskless prompts (no `--im`) and masked ones. The inpaint source defaults to the `--v` reference (so only the mask is required), and `--im` composes with `--i` (first-frame anchor) on the pure v2v path. `--im` is composed only on the pure v2v path; it is ignored (with a warning) on the audio-bearing `av_ic` / `video_ref_only_av` strategies.

| Flag | Default | Description |
|------|---------|-------------|
| `--im <path>` | — | Inpaint mask (image or video), resized to latent geometry. White = conditioned (kept clean) unless inverted |
| `--ii <0\|1>` | 0 | Invert: condition the complement of the mask (generate the masked region instead of keeping it) |
| `--it <float>` | 0.5 | Mask binarization threshold (strict `>`) |

```
A man in a park --v src.mp4 --i edited_frame0.png --im edit_mask.mp4 --ii 1 --it 0.5
```

##### IC-LoRA Arguments
<sub>[↑ contents](#table-of-contents)</sub>

| Argument | Default | Description |
|----------|---------|-------------|
| `--reference_downscale` | 1 | Spatial downscale factor for references (1=same res, 2=half) |
| `--reference_temporal_scale` | 1 | Temporal subsample factor for references (1=same frame rate, 2=half). Pass the same value used at caching |
| `--reference_frames` | 1 | Number of reference frames for V2V (images are repeated to fill this count) |
| `--ltx2_first_frame_conditioning_p` | 0.1 | Probability of also conditioning on the first target frame during training. No effect for single-frame (image) samples |
| `--ic_lora_ref_probability` | 1.0 | Probability the reference is applied each step (`v2v` / `av_ic` / `video_ref_only_av`). Below 1.0, the whole reference is randomly dropped that step so the model also learns to generate without it. Must be in `[0, 1]` |
| `--sample_include_reference` | off | Show reference side-by-side with generated output in sample videos |
| `--lora_target_preset v2v` | — | Targets attention + FFN layers (recommended for IC-LoRA) |

##### IC-LoRA Dataset Config Options
<sub>[↑ contents](#table-of-contents)</sub>

| Option | Type | Description |
|--------|------|-------------|
| `reference_directory` | string | Path to reference images/videos for IC-LoRA datasets (matched by filename stem) |
| `reference_directories` | array[string] | Optional multi-reference variant of `reference_directory`; use one entry per reference stream |
| `reference_cache_directory` | string | Output directory for cached reference latents |
| `reference_cache_directories` | array[string] | Optional multi-reference variant of `reference_cache_directory`; count must match `reference_directories` |
| `reference_frames` | int | Optional per-dataset override for `--reference_frames` during reference latent caching |

##### Notes
<sub>[↑ contents](#table-of-contents)</sub>

- **First-frame conditioning** (`--ltx2_first_frame_conditioning_p`): Randomly conditions on the first target frame in addition to the reference. Only applied during training; inference always denoises the full target. Has no effect for single-frame (image-only) samples — the code skips conditioning when `num_frames == 1` since there are no subsequent frames to generate.
- **Multi-frame references**: Supported but increase VRAM usage proportionally to the number of reference tokens.
- **Multi-reference datasets**: `av_ic` can consume multiple references directly from dataset TOML via `reference_directories` + `reference_cache_directories` (and the audio equivalents below). The list lengths must match. `--av_multi_ref` exposes that intent in training metadata/UI.
- **Multi-subject references**: The VAE compresses 8 frames into 1 temporal latent via `SpaceToDepthDownsample`, which pairs consecutive frames and averages their features. Subjects sharing the same 8-frame group are blended and lose individual identity. To keep N subjects separated, structure your reference video as: frame 1 = Subject A, frames 2–9 = Subject B (repeated 8×), frames 10–17 = Subject C (repeated 8×), etc. Total frames: `1 + 8×(N−1)`. Set `--reference_frames` to match. Frame 1 gets its own latent due to causal padding in the encoder; each subsequent 8-frame block produces one additional latent.
- **v2v is video-only**: the `v2v` IC-LoRA strategy requires `--ltx2_mode video` (audio-video mode is rejected for v2v reference conditioning). Audio-video reference IC-LoRA uses the `av_ic` / `video_ref_only_av` strategies (which require `--ltx2_mode av`) and `audio_ref_ic` (which requires `--ltx2_mode av` or `audio`).
- **Reference dropout** (`--ic_lora_ref_probability`): Below 1.0, the reference conditioning is randomly dropped for a fraction of training steps (a single batch-wide draw per step, since concatenation changes the sequence length), so the model also learns to generate without the reference. Default 1.0 always applies the reference. Only affects the reference-video strategies (`v2v` / `av_ic` / `video_ref_only_av`). Recorded as `ss_ic_lora_ref_probability` when < 1.
- **Downscale factor metadata**: Saved in LoRA safetensors as `ss_reference_downscale_factor`, plus `reference_downscale_factor` and `reference_spatial_scale_factor`, when factor != 1.
- **Temporal scale factor metadata**: Saved as `ss_reference_temporal_scale_factor` and `reference_temporal_scale_factor` when the factor != 1. Training and inference infer the actual scale from the cached/target latent frame counts, so the reference cache and target must keep the geometry the factor implies (`--reference_frames` = `8*k + 1`, factor dividing `k`, reference spanning the target length). Changing `--reference_temporal_scale` between cache runs re-encodes affected references even with `--skip_existing`.
- **Two-stage inference**: Not supported with V2V; a warning is emitted and the reference is ignored.

#### Audio-Reference IC-LoRA
<sub>[↑ contents](#table-of-contents)</sub>

Trains a LoRA using in-context audio-reference conditioning. Reference audio latents (clean, timestep=0) are concatenated with noisy target audio latents during training. Loss is computed only on the target portion. In AV mode the LoRA targets audio self/cross-attention, audio FFN, and bidirectional audio-video cross-modal attention layers; in audio-only mode the `audio` preset is auto-selected, which omits cross-modal layers that connect to the (dummy) video branch.

Supported modes:
- **`--ltx2_mode av`** — full audio-video model; trains both video and audio IC-LoRA layers.
- **`--ltx2_mode audio`** — audio-only mode; trains only audio layers (video is a dummy zero tensor). `--lora_target_preset audio` is auto-selected (cross-modal layers that affect the dummy video branch are omitted).

##### Recommended settings
<sub>[↑ contents](#table-of-contents)</sub>

The recommended audio-reference configuration uses the following settings. The trainer warns when the main audio-reference separation flags are off, and warns about first-frame conditioning only when it is effectively disabled in AV mode.

| Setting | Recommended value | Why |
|---------|-------------------|-----|
| `--ltx2_first_frame_conditioning_p` | `0.9` | Face identity comes from the first frame; voice identity comes from the reference LoRA. Without this, face identity is uncontrolled. AV mode only; no effect in audio-only mode. |
| `--audio_ref_use_negative_positions` | enabled | Clean positional separation between reference and target in RoPE space. |
| `--audio_ref_mask_cross_attention_to_reference` | enabled | Forces video to sync with target audio only (not reference). AV mode only. |
| `--audio_ref_mask_reference_from_text_attention` | enabled | Prevents reference audio from attending to text describing the target speech. |
| `--timestep_sampling` | `shifted_logit_normal` | Timestep distribution for audio-reference training. |
| `--network_dim` / `--network_alpha` | `128` / `128` | Recommended LoRA rank. |

> **Inference note**: Attention masks (`--audio_ref_mask_cross_attention_to_reference` and `--audio_ref_mask_reference_from_text_attention`) are **training scaffolding only**. They are automatically disabled during sampling/inference. The masks force the model to learn the attention separation during training; at inference the LoRA weights have already internalized it, so the masks are unnecessary.

##### How it works
<sub>[↑ contents](#table-of-contents)</sub>

1. Reference audio is encoded to latents and concatenated with noisy target audio along the temporal axis.
2. Reference tokens receive timestep=0 (no noise); target tokens receive the sampled sigma.
3. Loss is masked to exclude the reference portion — the model only learns to predict the target.
4. Three optional attention overrides control how the reference interacts with the rest of the model:
   - **Negative positions**: shifts reference tokens into negative RoPE time, creating clean positional separation from target tokens.
   - **A2V cross-attention mask**: blocks video from attending to reference audio (video syncs with target audio only).
   - **Text attention mask**: blocks reference audio from attending to text (reference provides identity, not content).

##### Step 1: Prepare Data
<sub>[↑ contents](#table-of-contents)</sub>

Organize training videos with matching reference audio files (same filename stem):

```
videos/                    reference_audio/
  speaker_001.mp4            speaker_001.wav    # reference clip for speaker_001
  speaker_002.mp4            speaker_002.flac
```

Reference audio files are matched to training videos by filename stem.

##### Step 2: Dataset Config
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

No special flags. Match the mode you are training:

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
<sub>[↑ contents](#table-of-contents)</sub>

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
  --lora_target_preset audio_ref_ic ^
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
  --ic_lora_strategy audio_ref_ic ^
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
<sub>[↑ contents](#table-of-contents)</sub>

Use `--ra <path>` in your sampling prompts file to specify the reference audio:

```
--ra reference_audio/speaker_001.wav A person speaking about nature --n blurry, low quality
--ra reference_audio/speaker_002.flac A woman laughing in a park
```

Reference audio latents are precached automatically when using `--precache_sample_latents` during latent caching.

##### Audio-Reference IC-LoRA Arguments
<sub>[↑ contents](#table-of-contents)</sub>

| Argument | Default | Description |
|----------|---------|-------------|
| `--ic_lora_strategy audio_ref_ic` | auto | Activates audio-reference IC-LoRA mode (auto-inferred from `--lora_target_preset audio_ref_ic`) |
| `--lora_target_preset audio_ref_ic` | — | Targets audio attn/FFN + bidirectional AV cross-modal layers |
| `--audio_ref_use_negative_positions` | off | Place reference audio in negative RoPE time for positional separation |
| `--audio_ref_mask_cross_attention_to_reference` | off | Block video from attending to reference audio tokens (AV mode only; no effect in audio-only mode) |
| `--audio_ref_mask_reference_from_text_attention` | off | Block reference audio from attending to text tokens (`av_ic`: currently unsupported and ignored) |
| `--audio_ref_identity_guidance_scale` | 0.0 | Identity guidance scale for `audio_ref_ic` sampling. When > 0, runs an extra forward pass without reference audio and amplifies the speaker-identity contribution (`audio_x0 += scale·(cond_with_ref − cond_without_ref)`). 0.0 disables it; reference suggests ~3.0 |

##### Audio-Reference Dataset Config Options
<sub>[↑ contents](#table-of-contents)</sub>

| Option | Type | Description |
|--------|------|-------------|
| `reference_audio_directory` | string | Path to reference audio files (matched by filename stem) |
| `reference_audio_directories` | array[string] | Optional multi-reference variant of `reference_audio_directory`; use one entry per reference stream |
| `reference_audio_cache_directory` | string | Output directory for cached reference audio latents |
| `reference_audio_cache_directories` | array[string] | Optional multi-reference variant of `reference_audio_cache_directory`; count must match `reference_audio_directories` |

##### Notes
<sub>[↑ contents](#table-of-contents)</sub>

- **Checkpoint**: requires an LTXAV checkpoint for both `--ltx2_mode av` and `--ltx2_mode audio`.
- **Bucket separation**: `separate_audio_buckets = true` keeps audio/non-audio items in separate batches (avoids shape mismatches in collation).
- **Attention masks are training-only**: `--audio_ref_mask_cross_attention_to_reference` and `--audio_ref_mask_reference_from_text_attention` are applied only during training. They are automatically disabled during sampling/inference (both set to `false` at validation). Negative position overrides are always applied.
- **`av_ic` limitation**: `--audio_ref_mask_reference_from_text_attention` is not currently supported in `av_ic` because the Modality API uses a 2D `context_mask`; the trainer warns and ignores this flag.
- **AV cross-attention modes**: `--av_cross_attention_mode both` is the default `av_ic` behavior. Use `a2v_only` for audio-to-video only, `v2a_only` for video-to-audio only, or `none` to disable AV cross-modal attention while keeping the rest of `av_ic` intact. All require `--ltx2_mode av`.
- **Multi-reference `av_ic`**: concatenates multiple reference latents before conditioning. `--av_multi_ref` does not change training behavior; it only records multi-reference intent in run metadata/UI.
- **`video_ref_only_av`**: requires `--ltx2_mode av`, uses reference video only, and keeps the audio branch target-only. Use it for identity/motion conditioning from video without requiring reference audio for every sample. Batches that carry cached target audio train both the video and audio branches; batches without cached audio latents train the reference-conditioned video branch only (same as regular av-mode training), so audio generation is supervised on the audio-bearing subset. Use `--audio_supervision_mode`/`--audio_supervision_min_ratio` to monitor the supervised-audio ratio.
- **First-frame conditioning**: for identity-sensitive AV IC-LoRA, `--ltx2_first_frame_conditioning_p 0.9` is the documented starting point. Without it, identity transfer from the target first frame is often weak. A warning is emitted in AV mode if `--ltx2_first_frame_conditioning_p` is effectively off (below ~0.01).
- `--ic_lora_strategy auto` (default) infers the strategy from `--lora_target_preset` via `infer_ic_lora_strategy_from_preset()`.

### Directional Training (A2V / V2A)
<sub>[↑ contents](#table-of-contents)</sub>

Trains a dedicated directional model by freezing one whole modality as clean conditioning (timestep 0, sigma 0, excluded from the loss) while the other is generated. Requires `--ltx2_mode av`. Off by default (`joint`).

| Flag | Default | Description |
|---|---|---|
| `--ltx2_train_direction` | `joint` | `joint` = normal joint AV denoising; `a2v` = freeze audio, generate video; `v2a` = freeze video, generate audio (foley). |

Both modality streams still run, so the frozen-clean one conditions the generated one through cross-attention. The frozen modality contributes no loss; its `loss_v` / `loss_a` logs as `n/a`.

This differs from **Cross-Task Synergy** (`--cts_lambda_*`): CTS adds directional losses *on top of* joint training (extra forwards per step; the model stays joint-capable), whereas `--ltx2_train_direction` *replaces* the main loss for a dedicated directional model (one forward per step).

Caveats:

- **AV mode required**: rejected for `--ltx2_mode video` / `audio`.
- **Standalone mode**: rejected together with IC-LoRA, `--self_flow`, `--tread`, Cross-Task Synergy, the G2D modality freezer (`--modality_freeze_*`), audio loss balancing (`--audio_loss_balance_mode`), `--dcr`, and `--tarp`.
- **`v2a` excludes video-side conditioners** because the whole video is frozen. First-frame conditioning (`--ltx2_first_frame_conditioning_p`, on by default at 0.1) is **auto-disabled with a warning** under `v2a` (it is redundant when the whole video is clean), so a plain `--ltx2_mode av --ltx2_train_direction v2a` runs as-is. Explicitly enabled video conditioners are rejected (disable one): `--ltx2_spatial_crop`, `--ltx2_inpaint_mask`, `--ltx2_extend_*`, `--video_anchor_training`, `--hfato`. `a2v` allows video conditioners (the video is generated).
- **`a2v` excludes `--independent_audio_timestep`**: audio is frozen at sigma 0, so a separately sampled audio timestep is discarded.
- **Training-only**: no inference behavior is added (inference-time A2V/V2A guidance is the separate `--video_modality_scale` / `--audio_modality_scale`).

### Latent Guides
<sub>[↑ contents](#table-of-contents)</sub>

Latent guides inject a per-sample reference latent directly into the video token stream — orthogonal to the IC-LoRA reference-video pathway. They are loaded from dataset directories (one stem-matched image per video sample), VAE-encoded once during latent caching, and applied automatically at training and inference time.

Two kinds, mirroring upstream Lightricks `VideoConditionByLatentIndex` / `VideoConditionByKeyframeIndex`:

- **`latent_idx`** — *token replacement* at a fixed frame slot. The guide latent overwrites tokens at `frame_idx` (and the loss is masked there, so the model only learns to denoise the remaining frames). `frame_idx=0` reproduces standard I2V conditioning; non-zero indices anchor a frame mid-video. Loss/timestep parity is preserved by re-masking the affected tokens as clean (`t=0`) per-token.
- **`keyframe`** — *token append* with custom positional encoding. A patchified keyframe latent is concatenated to the video token sequence. Positions are offset by `frame_idx` then divided by `frame_rate`. `frame_idx=-1` yields slightly-negative timestamps (the global-reference convention from upstream), making the keyframe visible to every output frame without claiming a specific slot. Predictions for the appended tokens are sliced off before loss.

  Strength is plumbed via per-token `denoise_mask = 1 - strength`, matching upstream `VideoConditionByKeyframeIndex`. The appended tokens get effective timestep `denoise_mask × sigma`: at `strength=1.0` this is `t=0` (clean conditioning); at `strength=0.0` this is `t=sigma` (the model sees the keyframe as pure noise, so it contributes no information). For scalar (per-batch) strength the latent itself is never modified — strength only modulates how "clean" the model perceives the keyframe to be. Per-sample strength tensors (used by Endpoint Keyframe Training) are an exception: when some samples in the batch have `strength<=0`, those samples' guide latent is replaced with `randn` noise before patchifying, to prevent leaking clean target-derived content into a fully-noisy mask slot.

Both can be set independently and per-dataset.

When a `latent_idx` guide and an intrinsic (`first_frame` / `spatial_crop` / `inpaint` / `extend`) condition the same frames, the guide takes precedence: it is pasted last and supplies the clean conditioning content for its slot (with the slot excluded from loss), so the intrinsic's target content is overridden in the overlapping frames. The trainer logs a one-time warning naming the overlapping intrinsic. `keyframe` guides append tokens instead of replacing a slot, so they never override an intrinsic.

#### Latent Guide Dataset Config Options
<sub>[↑ contents](#table-of-contents)</sub>

| Option | Type | Default | Description |
|---|---|---|---|
| `latent_idx_guide_directory` | string | — | Stem-matched guide images. Activates `latent_idx` for this dataset. |
| `latent_idx_guide_cache_directory` | string | — | Required when the source directory is set. |
| `latent_idx_guide_frame_idx` | int | `0` | Slot to replace. Training raises `ValueError` if `frame_idx < 0` or `frame_idx + T_guide > num_frames`. |
| `latent_idx_guide_strength` | float | `1.0` | Replacement strength. Values below `1.0` require `--ltx2_graded_conditioning` during training and use timestep `(1-strength) × sigma`; `0.0` disables the guide. |
| `keyframe_guide_directory` | string | — | Stem-matched global-reference images. Activates `keyframe`. |
| `keyframe_guide_cache_directory` | string | — | Required when the source directory is set. |
| `keyframe_guide_frame_idx` | int | `-1` | **Pixel-frame** index (NOT latent-frame). `-1` = global reference (slightly-negative timestamps, visible to all frames). For non-negative values, multiply the desired latent-frame by `VIDEO_SCALE_FACTORS.time` first — for LTX-2 (8× temporal VAE), latent frame `L` corresponds to `frame_idx = L × 8`. Example: anchor at the last latent frame of a 9-latent-frame video → `frame_idx = 64`. |
| `keyframe_guide_strength` | float | `1.0` | Per-token `denoise_mask = 1 - strength` → effective appended timestep = `(1-strength) × sigma`. `1.0` = clean conditioning; `0.0` = full noise / no contribution. Clamped to `[0, 1]`. |
| `keyframe_guide_extra_directories` | array[string] | — | Optional. Stack additional keyframes beyond the primary. |
| `keyframe_guide_extra_cache_directories` | array[string] | — | Cache directories for the extras (parallel to the directories list). |
| `keyframe_guide_extra_frame_idxs` | array[int] | — | Per-extra `frame_idx` values. |
| `keyframe_guide_extra_strengths` | array[float] | — | Per-extra `strength` values. |

The four `keyframe_guide_extra_*` arrays must all have the same length. The primary keyframe (above) must also be set.

Guide caches use the source image/video stem rather than a sampled video-window
stem. One independently encoded guide cache is therefore shared by all frame
windows selected from the same source video.

`strength` semantics differ between guide types and are NOT interchangeable:

- **`latent_idx_guide_strength`** is a **replacement-lock strength**. The guide latent overwrites tokens at the slot in-place. Training is binary by default; `--ltx2_graded_conditioning` enables continuous values with effective timestep `(1-strength) × sigma`.
- **`keyframe_guide_strength`** is an **append-guide strength**. The guide latent is appended as a new token block at the configured `frame_idx`; the original target tokens are still denoised normally. `strength` controls the appended block's `denoise_mask` (`1.0` = clean conditioning, `0.0` = effectively absent and the guide is dropped). Continuous values are accepted at both training and inference.

To force frame N pixels to *exactly* match a reference image, use `latent_idx`. To *guide* the model toward a reference, use `keyframe`.

Example multi-keyframe TOML:
```toml
keyframe_guide_directory = "/data/identity"
keyframe_guide_cache_directory = "/cache/identity"
keyframe_guide_frame_idx = -1
keyframe_guide_extra_directories      = ["/data/style"]
keyframe_guide_extra_cache_directories = ["/cache/style"]
keyframe_guide_extra_frame_idxs        = [5]
keyframe_guide_extra_strengths         = [0.7]
```

`ltx2_cache_latents.py` auto-encodes any guide directory it finds in the dataset config — no new CLI flags. Items differing in any guide-config detail (presence, count, `frame_idx`, `strength`) land in separate buckets, so each batch is shape-uniform.

#### IC-LoRA Compatibility Matrix
<sub>[↑ contents](#table-of-contents)</sub>

| `--ic_lora_strategy` | `--ltx2_mode` | `latent_idx` | `keyframe` |
|---|---|---|---|
| `none` | `video` / `av` | ✓ | ✓ |
| `v2v` | `video` | ✓ | ✓ |
| `av_ic` | `av` | ✓ | ✓ |
| `video_ref_only_av` | `av` | ✓ | ✓ |
| `audio_ref_ic` | `av` | ✓ | ✓ |
| `audio_ref_ic` | `audio` | n/a (no video target) | n/a |

`latent_idx` overwrites the noisy-target tensor before patchify, so it works on every branch that produces video tokens. `keyframe` token-append is wired through the `LTX2Wrapper.forward` path for the simple/audio-ref-only paths and via `build_keyframe_extension` for the v2v / av_ic / video_ref_only_av IC-LoRA branches; in all cases the appended timesteps are `(1 − strength) × sigma` and the predictions are sliced off before loss.

**Inference:** pure `v2v` composes token-appended `keyframe` guides through the
same `build_keyframe_extension` layout used during training, preserving
reference/target/keyframe ordering, positions, strengths, and target-only
prediction slicing. It also supports first-frame, inpaint, and `latent_idx`
conditioning. The audio-bearing reference strategies do not currently compose
token-appended keyframe guides at sample time.

### Reference target-frame routing

`reference_target_frame_ranges` is an opt-in extension to the existing
`reference_directories` interface. It limits which target-frame queries may
directly attend to each reference, without introducing another conditioning
schema:

```toml
reference_directories = ["data/reference_a", "data/reference_b"]
reference_cache_directories = ["cache/reference_a", "cache/reference_b"]
reference_target_frame_ranges = [[0, 17], [17, 33]]
```

Ranges use half-open pixel-frame coordinates `[start, end)` and are ordered
exactly like `reference_directories`. There must be one range per reference.
All datasets in a routed run must use the same ranges. The ranges are saved in
LoRA metadata and training-state checkpoints, and a mismatched resume is
rejected.

Pure-v2v inference accepts the same ordered layout:

```json
{
  "prompt": "A cinematic scene",
  "v2v_ref_paths": ["reference_a.mp4", "reference_b.mp4"],
  "reference_target_frame_ranges": [[0, 17], [17, 33]]
}
```

The standalone CLI equivalent is:

```bash
python ltx2_generate_video.py ... ^
  --reference_videos reference_a.mp4 reference_b.mp4
```

When a loaded LoRA contains `reference_target_frame_ranges` metadata, the
standalone script injects those ranges automatically and rejects a conflicting
prompt. A single-reference prompt may continue using `v2v_ref_path` and
`reference_target_frame_range`; `--reference_video` remains the compatible
single-reference CLI form.

Inference encodes the references in list order, concatenates them along the
same temporal latent axis as training, retains an individual token span for
each reference, and applies the same attention rule. Consequently, the
combined reference temporal geometry must satisfy the existing V2V temporal
scale constraint relative to the target; incompatible combinations fail
instead of silently changing positions. This feature does not replace or
rename official reference, keyframe, intrinsic, video, or audio conditioning
primitives.

Notes on edge cases:
- **`reference_downscale_factor`** (set on the dataset for v2v / av_ic / video_ref_only_av when ref-video resolution is lower than the target) is propagated into keyframe positions inside `build_keyframe_extension` so a downscaled keyframe carries spatial positions consistent with the ref-video. The simple path runs at full resolution and ignores this factor.
- **Inference image-resize on shape mismatch**: in `ltx2_inference.py:_denoise_loop`, both latent_idx and keyframe guide latents are bilinearly resized to the current denoising stage's spatial resolution if they don't already match (relevant for `--sample_two_stage` where stage 1 runs at half-resolution).
- **`--sample_i2v_token_timestep_mask`**: when set (default), inference only zeroes the token timestep at locked slots when `strength == 1.0`. With `strength < 1.0`, the per-token timestep follows `(1 − strength) × sigma` and the boolean mask is not applied.
- **Single-stage and two-stage sampling** both consume guides during sample-prompt previews.

#### Endpoint Keyframe Training
<sub>[↑ contents](#table-of-contents)</sub>

Optional training-time CLI flags that extract first / last / random-interior latent frames of the target video and append them as clean keyframe tokens, without requiring per-item keyframe images on disk. Composes with any `--ic_lora_strategy` (the endpoint guides are appended to the same `keyframe_guides_for_options` list that external keyframes use).

| Flag | Default | Description |
|---|---|---|
| `--keyframe_endpoint_training` | off | Master enable. All flags below are no-ops when this is unset. |
| `--keyframe_first_frame_p` | `1.0` | Per-sample probability of appending the first latent frame as a clean keyframe at `frame_idx=0` (independent Bernoulli per item in the batch). |
| `--keyframe_last_frame_p` | `1.0` | Per-sample probability of appending the last latent frame at `frame_idx=(T−1) × VIDEO_SCALE_FACTORS.time` (pixel-frame units; for LTX-2's 8× temporal VAE this is `(T−1) × 8`). |
| `--keyframe_random_interior_p` | `0.0` | Per-sample probability of appending random interior latent frames as keyframes. Interior indices are shared across the batch; only the dropout decision is per-sample. |
| `--keyframe_max_random_interior` | `0` | Maximum number of random interior latent frames to append per batch when any sample triggers. Sampled without replacement from `[1, T−2]`, sorted ascending. |

When the master flag is on, each per-frame probability is sampled independently **per sample** within the batch (Bernoulli; `1.0` always fires, `0.0` never fires). Within one batch, some samples may receive a given endpoint guide while others do not — the guide is appended uniformly (required for tensor packing) but the per-sample denoise_mask is `0` (clean) for samples that won the flip and `1` (no effect) for those that didn't. For losers (samples that did not win the flip), the appended guide latent is replaced with random noise *before* patchifying, so the model sees noise tokens with `denoise_mask=1` rather than clean target-derived content tagged as fully noisy — this prevents a leak of the supervision target through the appended-token stream. Endpoint guides stack with any external keyframes from `keyframe_guide_directory` (external first, endpoints appended after).

Example: train an interpolation LoRA from raw videos, both endpoints always present, no interior augmentation:
```
--keyframe_endpoint_training \
--keyframe_first_frame_p 1.0 \
--keyframe_last_frame_p 1.0
```

Example: same but with up to 2 random interior keyframes 30% of the time, for a model that learns to use keyframes at arbitrary positions:
```
--keyframe_endpoint_training \
--keyframe_random_interior_p 0.3 \
--keyframe_max_random_interior 2
```

Caveats and tradeoffs:

- **Distribution match**: endpoint keyframes are sliced directly from the encoded video latent, not from a separately VAE-encoded still image. The first latent frame of LTX-2's 8× causal VAE encodes essentially pixel-frame 0 (causal_fix anchors it), so first-frame extraction is close to a still-image encode. The last and interior latent slices represent ~8 pixel-frames each of motion context, so they carry temporal information a single still image would not. To match the inference distribution exactly when keyframes will come from images at sample time, prefer the dataset-driven `keyframe_guide_directory` workflow with images that are encoded individually via the same VAE, OR train on still-image-derived latents and use endpoint extraction only as augmentation.
- **Soft conditioning, not hard locks**: keyframes are appended tokens with `denoise_mask = 1 - strength`. The original target latent at the corresponding frame is still denoised normally. The model learns to be guided by the keyframe; it does not have a hard constraint to reproduce it pixel-exact. For exact endpoint preservation (image-to-video with frame 0 fixed), use a `latent_idx` guide (token replacement at that slot) instead of (or in addition to) keyframe append.
- **Distribution under defaults**: defaults `first_p=1.0, last_p=1.0, interior_p=0.0` train two-ended interpolation. Single-anchor i2v (set `last_p=0`) and last-anchor extension (set `first_p=0`) are out-of-distribution unless you train with those probabilities directly.
- **`strength=0` keyframes are skipped entirely** (they would otherwise add tokens to attention with no useful signal). At inference, this means a `--gk` guide with `:0.0` is equivalent to omitting it.

#### Video Anchor Training
<sub>[↑ contents](#table-of-contents)</sub>

Optional training-time augmentation for workflows that use clean video frames as anchors. During some training samples, selected target frames are kept as known frames while the model trains on the rest of the video.

| Flag | Default | Description |
|---|---|---|
| `--video_anchor_training` | off | Master enable. When unset, generated commands do not include the video-anchor flags. |
| `--video_anchor_probability` | `0.5` | Per-sample probability of applying video-anchor training. |
| `--video_anchor_count` | `1` | Number of random anchors to add per sample when the strategy includes random anchors. |
| `--video_anchor_strategy` | `endpoints_random` | `endpoints` keeps first/last frames only, `random` samples anchors uniformly, and `endpoints_random` combines both. |
| `--video_anchor_strength` | `1.0` | Anchor strength. Lower values require `--ltx2_graded_conditioning` and use timestep `(1-strength) × sigma`; `0.0` disables anchors. |
| `--ltx2_graded_conditioning` | off | Enables partial-strength `latent_idx` guides and video anchors. Existing binary conditioning is unchanged when off. |
| `--ltx2_causal_temporal_attention` | off | Restricts video self-attention to the current and earlier video times for causal/streaming-oriented training. Requires `--sdpa`, may reduce throughput, and leaves normal bidirectional attention unchanged when off. |
| `--ltx2_soft_av_alignment` | off | Biases A2V and V2A cross-attention toward tokens with nearby timestamps. Requires AV mode and `--sdpa`; off leaves cross-attention unchanged. |
| `--ltx2_soft_av_alignment_sigma` | `0.5` | Gaussian alignment width in seconds. Smaller values impose tighter temporal locality. |

Applicable when the target workflow benefits from anchor-like conditioning, such as first/last-frame control, video-to-video refinement, reference-guided training, or experiments where the model should see fixed in-clip frames while learning the surrounding motion. Not a general quality switch for every run.

Suggested starting point:
```
--video_anchor_training \
--video_anchor_probability 0.25 \
--video_anchor_count 1 \
--video_anchor_strategy endpoints_random
```

Caveats and tradeoffs:

- **Video target required**: `--video_anchor_training` is rejected for `--ltx2_mode audio` and audio-only checkpoints because there are no video target latents to anchor.
- **Random strategy needs anchors**: `--video_anchor_strategy random` requires `--video_anchor_count >= 1`; use `endpoints` if you only want first/last-frame anchors.
- **No inference behavior is added**: this is training-only.
- **No guaranteed quality gain**: evaluate against your target prompts and sampling workflow before using it as a default.

#### Sample Prompt Flags
<sub>[↑ contents](#table-of-contents)</sub>

| Flag | Meaning |
|---|---|
| `--gl frame_idx:path[:strength]` | `latent_idx` guide for this prompt. Multiple allowed. |
| `--gk frame_idx:path[:strength]` | `keyframe` guide for this prompt. Multiple allowed. |

Guides take effect on both single-stage and two-stage sampling paths.

### Conditioning representation
<sub>[↑ contents](#table-of-contents)</sub>

LTX-2 conditions per token: every token in the shared audio-video latent stream carries its own latent value, timestep, 3D position (RoPE), modality, and attention/loss membership. The recipe sets these fields — an intrinsic condition marks tokens clean (timestep 0, held value, dropped from the loss); a reference condition concatenates reference tokens at their own positions; directional training freezes a modality's tokens at timestep 0. There is no fixed mode; a training scheme is a configuration of per-token fields.

External editing and control work builds on the same representation:
- **TIDE** ([arXiv 2606.08260](https://arxiv.org/abs/2606.08260)) — in-context editing via per-token task embeddings that label target, source, and reference tokens in one stream.
- **MMControl** ([arXiv 2604.19679](https://arxiv.org/abs/2604.19679)) — multi-modal control on the LTX-2 19B backbone: temporal reference concatenation with a binary keep/generate mask, plus added bypass branches for depth, pose, and audio.

## Slider LoRA Training
<sub>[↑ contents](#table-of-contents)</sub>

> Slider LoRA training is based on the [ai-toolkit](https://github.com/ostris/ai-toolkit) implementation by ostris, adapted for LTX-2.

Slider LoRAs learn a controllable direction in model output space (e.g., "detailed" vs "blurry"). At inference, you scale the LoRA multiplier to control the effect strength and direction: `+1.0` enhances, `-1.0` erases, `0.0` is the base model, and values like `+2.0` or `-0.5` work too.

**Script:** `ltx2_train_slider.py`

Three modes are available:

| Mode | Input | Use Case |
|------|-------|----------|
| `text` | Prompt pairs only (no dataset) | Sliders from text prompt pairs, no images needed |
| `reference` | Pre-cached latent pairs | Sliders from paired positive/negative image, video, or audio samples |
| IC-slider (`mode = "ic_reference"`) | Paired target caches + shared reference caches | Slider training under shared `v2v` reference conditioning |

### Text-Only Mode
<sub>[↑ contents](#table-of-contents)</sub>

Learns a slider direction from positive/negative prompt pairs. No images or dataset config needed.

#### Slider Config (`ltx2_slider.toml`)
<sub>[↑ contents](#table-of-contents)</sub>

```toml
mode = "text"
guidance_strength = 1.0
anchor_strength = 1.0
batch_all_targets = false   # true = train on every target each step (averaged)
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]

[[targets]]
positive = "extremely detailed, sharp, high resolution, 8k"
negative = "blurry, out of focus, low quality, soft"
target_class = ""      # empty = affect all content
weight = 1.0

[[anchors]]
prompt = "a portrait of a person"

[[anchors]]
prompt = "a landscape"
```

- `guidance_strength`: Scales the directional offset applied to targets. Higher values = stronger direction signal but may overshoot.
- `target_class`: The conditioning prompt used during training passes. Empty string means the slider affects all content regardless of prompt. Set to e.g. `"a portrait"` to restrict the slider's effect to a specific subject.
- `weight`: Per-target loss weight. Controls relative emphasis when training multiple directions simultaneously.
- `sample_slider_range`: Multiplier values used for preview samples during training.

Multiple `[[targets]]` blocks can be defined to train several directions at once (e.g., detail + lighting).

- `batch_all_targets`: Text mode only (default `false`). When `false`, one target is picked at random each step. When `true`, **every** target is processed in the same step and their gradients are averaged (each target's loss scaled by `1/N`) into a single optimizer update. Averaging yields a lower-variance, more context-general direction — the signal shared across targets reinforces while context-specific noise partially cancels — at roughly `N x` the per-step compute. This is **not** equivalent to simply training longer: doubling steps applies sequential, separately-clipped updates that can interfere, whereas batching reconciles all targets in one move. If anchors are configured, the anchor loss is scaled by the same `1/N`, so anchor magnitude/behaviour is unchanged versus the non-batched path.

##### Anchors (concept preservation)

Optional `[[anchors]]` blocks define concepts that should remain unchanged by the slider. For each anchor prompt, the frozen base model's prediction is computed once per step, and the LoRA is constrained to reproduce that same prediction at *both* slider multipliers (`+1` and `-1`). This prevents the slider from drifting on unrelated content — the classic concept-slider preservation technique. (Credit: [phazei](https://github.com/phazei) — [PR #87](https://github.com/AkaneTendo25/musubi-tuner/pull/87))

The per-step anchor loss (a plain MSE between the LoRA and frozen-base predictions) is intrinsically spiky in flow-matching velocity space: on ~90% of steps the LoRA barely moves the anchor prediction (loss ~1e-4), but on a small fraction of steps it deviates a lot. An offline probe over real predictions confirmed this spikiness is **not** a removable per-step scale factor — every per-step normalizer tried (variance-normalize, cosine, relative-MSE) still left a 1000-7000x dynamic range, because the variation lives in the genuine deviation, not in a divisible scale. Instead, each step's anchor loss is **capped at a multiple of its running median** (`anchor_cap_mult`), which bounds the rare gradient-norm spikes while leaving the typical steps untouched. This mirrors the spirit of Min-SNR-γ clamping. The cap is applied gradient-preservingly (the loss is scaled by a detached ratio), so the optimizer still moves in the anchor's direction — just not with explosive magnitude.

> **Important — `loss/anchor` magnitude is NOT a measure of anchor impact.** The anchor loss value is typically tiny (~1% of `loss/average`) because it is a *squared* near-zero residual. Its *gradient*, however, is a non-squared, consistently-directed pull on the LoRA weights every step, so even a tiny loss value meaningfully constrains training. In practice, adding an anchor noticeably raises the steady-state direction loss (e.g. ~0.0048 → ~0.0078 on the skin-tone test) because the LoRA spends capacity preserving the anchor instead of maximizing the slide. That higher direction loss is the *cost of preservation*, working as designed — it does **not** mean training is broken. Judge anchor effectiveness from samples (does the anchor concept hold at ±2 while the slider direction still works), never from the `loss/anchor` number.

- `[[anchors]]` / `prompt`: A text prompt describing a concept to preserve. Add as many blocks as needed.
- `anchor_strength`: Weight on the anchor preservation loss (default `1.0`). Raise it if you observe drift on the anchor concept; lower it if the slider direction is being suppressed. Because the anchor's gradient pull is not proportional to its small loss value, treat this as an empirical knob tuned from samples.
- `anchor_cap_mult`: Per-step spike cap, as a multiple of the running median anchor loss (default `5.0`, set `0` to disable). Lower values (3-5) clamp spikes harder and keep grad_norm calmer; higher values let more of the raw signal through. The cap only affects rare spike steps, so it stabilizes grad_norm **without** weakening the anchor's steady-state preservation pull (which comes from the ~90% of normal steps). The `loss/anchor` TensorBoard series reflects the post-cap value.

Each anchor adds one extra row to both the no-grad reference forward pass and each gradient pass, so keep the anchor count small (typically 1-3). **Anchors apply to `text` mode only** — `reference` and `ic_reference` modes ignore them, since paired positive/negative examples already largely self-regularize the LoRA against unrelated drift.

> **Choosing anchors:** A good anchor is a concept *adjacent to but not on* the slider's axis. For a slider that changes an attribute of people (e.g. skin tone), avoid anchors where the model will inject a person — "a city street" or "a t-shirt" will spawn a person whose attribute then sits on the slider axis and fights training. People-free nature/landscape prompts (e.g. `"a mountain valley at sunset, forest and rocks, no people"`) are the safest anchors for person-attribute sliders.

#### Example Command (Text-Only)
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

| Argument | Default | Description |
|----------|---------|-------------|
| `--slider_config` | (required) | Path to slider TOML config file |
| `--latent_frames` | 1 | Number of latent frames (1=image, >1=video) |
| `--latent_height` | 512 | Pixel height for synthetic latents |
| `--latent_width` | 768 | Pixel width for synthetic latents |
| `--guidance_strength` | (from TOML) | Override `guidance_strength` from config |
| `--sample_slider_range` | (from TOML) | Override as comma-separated values, e.g. `"-2,-1,0,1,2"` |

All standard training arguments (`--fp8_base`, `--blocks_to_swap`, `--gradient_checkpointing`, etc.) work the same as regular training. `--dataset_config` is not needed for text-only mode.

### Reference Mode
<sub>[↑ contents](#table-of-contents)</sub>

Learns a slider direction from paired positive/negative image, video, or audio examples. Requires pre-cached latents.

#### Step 1: Prepare Paired Data
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

##### Slider Config (`ltx2_slider_reference.toml`)
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

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

For video reference sliders, `--ltx2_first_frame_conditioning_p` also works here. On multi-frame samples it fires stochastically per step with this probability (default 0.1), anchoring frame 0 as conditioning-only and excluding it from the loss, which is useful when positive/negative pairs share the same start frame and differ mainly in motion. It has no effect for text-only sliders or single-frame reference samples.

##### Audio Reference Sliders
<sub>[↑ contents](#table-of-contents)</sub>

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
  --output_dir output --output_name audio_reference_slider ^
  --slider_config ltx2_slider_audio_reference.toml
```

Notes:
- Audio reference sliders are supported in `--ltx2_mode audio`.
- The trainer uses the same paired positive/negative audio latents plus shared text cache, and masks loss with cached `audio_lengths`.
- `--lora_target_preset audio` is recommended; if omitted, the slider trainer selects it automatically for audio reference sliders.
- `--ltx2_first_frame_conditioning_p` has no effect for audio sliders.

### IC-slider
<sub>[↑ contents](#table-of-contents)</sub>

Trains a slider from paired positive/negative target latents under a shared cached visual reference. Internally this mode reuses the existing `v2v` IC path.

#### Slider Config (`ltx2_slider_ic_reference.toml`)
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

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
<sub>[↑ contents](#table-of-contents)</sub>

- **Start small**: `--network_dim 8` or `16` with `--max_train_steps 200-500` is usually sufficient.
- **Monitor loss**: Loss usually trends downward once training is stable. If it diverges, reduce `--learning_rate`.
- **Preview samples**: Add `--sample_prompts sampling_prompts.txt --sample_every_n_steps 50` to generate previews at each slider strength during training. Requires `--gemma_root` for text encoding. For audio sliders, also use `--sample_audio_only`.
- **Guidance strength**: For text-only mode, the default is `1.0`. Values of `2.0-3.0` increase direction strength but may reduce convergence stability.
- **Multiple targets**: Text-only mode supports multiple `[[targets]]` blocks. Each step randomly selects one target, so all directions get trained evenly.
- **Inference**: Use the trained LoRA with any multiplier value. Positive multipliers enhance the positive attribute, negative multipliers enhance the negative attribute. Values beyond `[-1, +1]` extrapolate the effect.

---

## Reinforcement-Learning Post-Training (RL-LoRA)
<sub>[↑ contents](#table-of-contents)</sub>

**Scripts:** `ltx2_cache_rollouts.py` (Phase A), `ltx2_train_rl.py` (Phase B), `ltx2_rl_rounds.py` (round-loop driver running both).

> [!WARNING]
> RL post-training for video generation is a young area — the methods this implementation builds on were published in 2025 ([Flow-GRPO](https://arxiv.org/abs/2505.05470), [DanceGRPO](https://arxiv.org/abs/2505.07818)) — and it inherits the long-documented practical realities of RL in general: results are sensitive to configuration (reward choice and weights, group size, learning rate, number of rounds); a reward is *maximized*, not *understood*, so it can be exploited instead of satisfied (reward hacking); training can pass its optimum and degrade in later rounds (pick the round by held-out score, not the last one); and a reward that barely separates the samples within a group produces noise, not signal. Expect to iterate on the setup rather than to get the target behavior from a single run.

### Overview
<sub>[↑ contents](#table-of-contents)</sub>

A LoRA trained the supervised way needs examples of the target result. RL post-training needs only a **score**. The model generates candidate videos, each one gets a number from a reward function, and training pushes the model toward what scores higher. It applies to properties you can *measure* but cannot *collect a dataset of*: human preference, physical plausibility, audio-video sync, prompt adherence — any metric you can compute on a generated sample (see [Writing a custom reward](#writing-a-custom-reward)). The output is an ordinary LoRA file — this document calls it an **RL-LoRA**: a LoRA whose weights come from reward scores rather than from a supervised dataset. The name describes how the weights were obtained, not the file format; it loads anywhere a regular LTX-2 LoRA does.

The offline loop samples `K` rollouts per prompt, scores them, computes group-relative advantages as `(reward - group_mean) / (group_std + 1e-4)`, and applies the selected objective. For `nft`, `rwr`, and `ppo`, `--nft_kl_beta` weights a mean-squared difference between policy and frozen-reference predictions. The flag and `kl` log key are historical names; this implementation does not calculate a KL divergence for that term.

The starting point is either a fresh LoRA (omit `--network_weights`) or an existing LoRA. A group whose `K` samples receives equal scores has zero normalized advantage and is skipped. The effect of training depends on the reward, generated samples, objective, and optimization settings; no capability or quality outcome is guaranteed.

**This implementation runs the loop as offline rollouts** — two separate processes instead of one:

- **Phase A** (`ltx2_cache_rollouts.py`) — data collection and scoring. It generates `K` rollouts per prompt with the current LoRA, runs the selected rewards, and writes a disk cache plus the `old` snapshot LoRA that generated it. It does **not** update weights.
- **Phase B** (`ltx2_train_rl.py`) — optimisation. It replays that exact cache, checks that the warm-start LoRA matches the cache's `old` snapshot, and applies the update rule (`--rl_loss`). In the offline path it does **not** call reward models again; it trains from the scores stored by Phase A.

In the offline workflow, rollout generation/scoring and Phase-B training run as separate processes, so their peak allocations are not simultaneous. This does not guarantee that either phase fits a particular GPU; memory depends on the model, precision, reward implementation, media decode, sampling settings, and training configuration. The separate phases also allow:

- the two phases can run on different GPUs, different machines, or different days;
- one compatible cache can be replayed for `nft`/`rwr`/`dpo` objective changes; `ppo` requires a cache recorded with Phase-A `--rl_sde_sampler`;
- rollouts and their scores sit on disk, so you can inspect what the reward is actually rewarding before spending any training compute;
- an interrupted run loses at most one round — every round's cache and LoRA are ordinary files.

Media decoding follows each reward's declared inputs. Audio-only rewards such
as `audio_energy`, CLAP, and Audiobox decode the generated audio without
decoding video pixels through the video VAE. Video and synchronization rewards
still request video decoding when they declare `video` or `video_file`.

The cache is tied to the snapshot that generated it; Phase B refuses a mismatched warm-start LoRA. The primary mode is a video LoRA; `--ltx2_mode av` generates video + audio and decodes both (audio via a subprocess vocoder) so audio/sync rewards can score (the file-based readers have dependency fallbacks — e.g. the stdlib wav module for audio when torchcodec is absent). An inline single-process mode (`--rl_online`) also exists, but it holds the generator, the rewards, and the training step on one GPU at once.

You configure three independent things; everything else has defaults:

1. **Reward** (`--reward_fn`) — what to optimise: the composable [reward zoo](#the-reward-zoo).
2. **Update rule** (`--rl_loss`) — `nft` (default), `rwr`, `dpo`, `ppo` turn GRPO advantages into gradients; `refl` instead backprops a differentiable reward directly — see [Update rules](#update-rules).
3. **The round loop** — how many generate→train rounds, at what learning rate — see [How to use](#how-to-use) and [Tips](#tips-1).

The design rests on four pieces:

- **GRPO group-relative advantages** (developed by DeepSeek) — the learning signal comes from comparing the `K` samples within each group, so no value/critic model has to be trained or held in memory. The advantage scheme is always on and is not itself an update rule: the full DeepSeek "GRPO algorithm" is these advantages plus a PPO-clip loss — `--rl_loss ppo` here — while the default `nft` applies the same advantages through a negative-aware update;
- **offline rollouts** — sample generation and reward scoring are decoupled from the training step, and rollouts are reusable files on disk;
- **interchangeable update rules** over one cached-rollout contract — `nft` (negative-aware update, after DiffusionNFT), `rwr`, Diffusion-DPO, and trajectory-faithful PPO/DDPO can be swapped and compared on identical rollouts;
- a **composable black-box reward zoo** — any computable per-sample metric can drive training, alone or in a weighted mix, with no paired dataset required.

Algorithm papers are listed under [References](#references).

### How to use
<sub>[↑ contents](#table-of-contents)</sub>

**1. Prerequisites.** The same LTX-2 checkpoint and Gemma as supervised training. No dataset is needed. Pick a reward from [the zoo](#the-reward-zoo) — install the extra deps listed at the top of its file (if any) and download its checkpoint (if model-backed) — or write your own metric and pass it via `--reward_plugins` (see [Writing a custom reward](#writing-a-custom-reward)).

**2. Write a prompt file** (`rl_prompts.txt`) — 8–16 diverse prompts in the domain you are training, one per line.

**3. Phase A — generate + score rollouts, write the cache:**

```bash
python src/musubi_tuner/ltx2_cache_rollouts.py ^
  --ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
  --gemma_root /path/to/gemma ^
  --fp8_base --fp8_scaled --blocks_to_swap 10 ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --network_weights output/my_lora.safetensors ^
  --rl_prompts rl_prompts.txt ^
  --rl_group_size 8 ^
  --reward_fn "hpsv3:1.0" ^
  --reward_args checkpoint_path=/models/hpsv3 config_path=/models/hpsv3/config.json ^
  --rl_rollout_cache output/rollouts_r0 ^
  --rl_save_old_lora output/old_snapshot_r0.safetensors ^
  --seed 42
```

`--rl_save_old_lora` writes the fp32 `old` snapshot that generated this cache; load it as `--network_weights` in Phase B so `default` starts exactly equal to `old` (the snapshot-hash invariant). `--network_weights` here is the LoRA being refined — omit it to start from the bare base model.

**4. Phase B — replay the cache, update the LoRA:**

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/ltx2_train_rl.py ^
  --ltx2_checkpoint /path/to/ltx-2.3.safetensors ^
  --gemma_root /path/to/gemma ^
  --fp8_base --fp8_scaled --blocks_to_swap 10 --gradient_checkpointing ^
  --network_module networks.lora_ltx2 ^
  --network_dim 16 --network_alpha 16 ^
  --network_weights output/old_snapshot_r0.safetensors ^
  --rl_rollout_cache output/rollouts_r0 ^
  --rl_loss nft ^
  --learning_rate 2e-4 --optimizer_type AdamW8bit --max_grad_norm 1.0 ^
  --output_dir output --output_name my_lora_rl_r0 ^
  --seed 42
```

The result, `output/my_lora_rl_r0.safetensors`, is a LoRA checkpoint and can be used as the next round's warm start. With `--logging_dir`, per-step loss/policy and historically named `kl` scalars are written to TensorBoard; for `nft`, `rwr`, and `ppo`, that scalar is the reference-prediction MSE described above.

Swap the update rule with `--rl_loss rwr|dpo` — nothing else changes; the same cache is reused. **`--rl_loss ppo`** (trajectory-faithful DDPO) instead needs a cache built with Phase-A `--rl_sde_sampler` (records the per-step trajectory):

```bash
# Phase A — add --rl_sde_sampler so the per-step SDE trajectory is cached (CFG stays off):
python src/musubi_tuner/ltx2_cache_rollouts.py ... --reward_fn "hpsv3:1.0" ^
  --rl_sde_sampler --rl_sde_eta 1.0 ^
  --rl_rollout_cache output/rollouts_ppo_r0 --rl_save_old_lora output/old_ppo_r0.safetensors

# Phase B — trajectory-faithful PPO on that cache:
accelerate launch ... src/musubi_tuner/ltx2_train_rl.py ... --network_weights output/old_ppo_r0.safetensors ^
  --rl_rollout_cache output/rollouts_ppo_r0 --rl_loss ppo --rl_sde_eta 1.0 --ppo_clip_eps 0.2
```

**5. The round loop.** RL proceeds in rounds: *generate → train → warm-start the next round from the new LoRA*. `ltx2_rl_rounds.py` runs the whole loop from one command — pass it the union of the Phase A and Phase B arguments (each flag is forwarded to the phase that owns it) plus `--rl_rounds`:

```bash
python src/musubi_tuner/ltx2_rl_rounds.py <the Phase A and Phase B arguments from above> ^
  --rl_rounds 10 ^
  --output_dir output --output_name my_lora_rl ^
  --rl_heldout_prompts heldout_prompts.txt   # optional evaluation/model selection
```

Omit `--rl_rollout_cache` / `--rl_save_old_lora` — the driver sets them per round and wires the warm-start/`old`-snapshot chain itself; `--network_weights` (optional) is the LoRA round 1 starts from. Rounds land in `output/my_lora_rl_rounds/round_NN/` (cache, `old` snapshot, the round's LoRA), and `progress.log` collects every `ROUND_REWARD`; the same per-round train/held-out reward curves are also written as TensorBoard scalars to `output/my_lora_rl_rounds/tb`. Re-running the same command resumes: finished rounds are skipped. `--rl_heldout_prompts` is optional: it does not affect the loss, but scores the starting point and each round's LoRA on fixed-seed prompts that were not used for training, so you can detect reward hacking/collapse and pick the round at the held-out peak (see [Tips](#tips-1)). `--rl_delete_round_caches` reclaims disk after each round.

The round driver is the recommended automated path. After the initial command, it repeats this chain without user intervention:

```text
round N Phase A: generate and score rollouts with the current LoRA, save cache + old.safetensors
round N Phase B: train from that exact old.safetensors on that exact cache, save output_name_rNN.safetensors
round N+1: use output_name_rNN.safetensors as the warm start for the next rollout generation
```

Do not pass a per-round cache path to `ltx2_rl_rounds.py`; doing so is managed internally so the cache, `old` snapshot, and warm-start LoRA cannot drift apart. If the process stops, run the same command again: completed round LoRAs are skipped, and the next missing round continues from the latest completed LoRA.

Driving the phases manually works too. Because `old` is frozen per cache, advance the behavior policy by regenerating the cache rather than training many passes over a stale one:

```text
round 0:  Phase A (--network_weights my_lora.safetensors    --rl_save_old_lora old_r0)  -> cache_r0
          Phase B (--network_weights old_r0 --rl_rollout_cache cache_r0)                -> my_lora_rl_r0
round 1:  Phase A (--network_weights my_lora_rl_r0           --rl_save_old_lora old_r1) -> cache_r1
          Phase B (--network_weights old_r1 --rl_rollout_cache cache_r1)                -> my_lora_rl_r1
...
```

Any round's Phase-B output is an RL-LoRA you can ship. If you enabled held-out scoring, pick the held-out peak; otherwise choose by visual review and any external evaluation you trust (see [Tips](#tips-1)).

### Update rules
<sub>[↑ contents](#table-of-contents)</sub>

`nft`/`rwr`/`dpo` consume the **same** cached rollouts, the same GRPO advantages, and the same three-forward `states` dict (`fwd`/`old`/`ref`/`x0`); only the final loss differs. `ppo` (trajectory-faithful DDPO) instead reads the per-step trajectory recorded by Phase-A `--rl_sde_sampler`. `refl` is different in kind — it is not policy-gradient and uses no cache or advantages; it backprops a differentiable reward straight through the sampler (`ltx2_refl.py`). Switch with `--rl_loss`:

| `--rl_loss` | What it does | Key flags |
|---|---|---|
| `nft` *(default)* | Negative-aware fine-tuning: reinforce high-advantage samples, push away from low ones. | `--nft_beta_mix`, `--nft_adv_clip_max` |
| `rwr` | Advantage-weighted regression: pull each sample toward its own clean `x0`, weighted by advantage. | `--rwr_temperature` |
| `dpo` | Diffusion-DPO on each group's best vs worst sample (preference; reward only ranks). | `--dpo_beta` |
| `ppo` | Trajectory-faithful DDPO: exact per-step importance ratio. Needs a Phase-A `--rl_sde_sampler` cache; run with CFG off. | `--ppo_clip_eps`, `--rl_sde_eta` |
| `refl` | Online differentiable-reward update. It performs one differentiable final denoising step from a re-noised detached rollout latent; AV training is explicitly enabled with `--refl_av`. | `--refl_av`, `--refl_renoise_samples`, `--refl_reward_weight`, `--nft_kl_beta` (reference-x0 MSE coefficient) |

`nft`/`rwr`/`ppo` and `refl` use `--nft_kl_beta` (default `1e-4`) for a frozen-reference MSE term; `dpo` does not use it. For `refl`, `--refl_grad_steps` accepts only `1`; values greater than one are rejected because multi-step differentiable denoising is not implemented. `--refl_renoise_samples M` averages `M` independent one-step estimates. `latent_energy` operates directly on the denoised video latent; `pixel_sharpness` requires `--vae` and backpropagates through the frozen video decoder. `audio_energy` backpropagates through the frozen audio decoder and vocoder when both `--ltx2_mode av` and `--refl_av` are set. Use BF16 decoders; FP16 video-decoder backward can underflow and fail to update the adapter.

### The reward zoo
<sub>[↑ contents](#table-of-contents)</sub>

Rewards are **composable plugins** (`ltx2_rewards/`). Each plugin declares three things: a `score(samples) -> ([float per sample], info)` method (always higher-is-better — any inversion such as desync→`1/(1+d)` is applied inside `score`), an explicit **`route`** (`video` | `audio` | `sync`), and a **`needs`** set of per-sample inputs (e.g. `{"video"}`, `{"audio_waveform"}`) that tells the generator which media to decode for scoring.

Select rewards with `--reward_fn "name:weight,name2:weight2,..."` (a bare `name` defaults to weight `1.0`). Built-in user-facing rewards:

| Reward | Route | What it scores | Source |
|--------|-------|----------------|--------|
| `hpsv3` | video | Frame-quality / human-preference (vendored Qwen2-VL) | MizzenAI HPSv3, via OmniNFT reward series |
| `videoreward` | video | VideoAlign visual-quality / motion-quality / text-alignment | KwaiVGI VideoAlign, via OmniNFT reward series |
| `videoscore2` | video | VideoScore2 VLM judge (VQ / TA / PC); `--reward_args dims=pc` selects its **Physical/common-sense Consistency** head | TIGER-Lab VideoScore2 |
| `iqa_quality` | video | No-reference frame IQA detail/quality via IQA-PyTorch; default `topiq_nr` | TOPIQ/LIQE/MUSIQ/etc. through IQA-PyTorch |
| `anti_noise` | video | Guardrail for detail rewards: penalizes flat-area speckle and temporal high-frequency flicker | in-repo |
| `clap` | audio | CLAP audio–text similarity | LAION CLAP, via OmniNFT reward series |
| `audiobox` | audio | Audiobox-Aesthetics audio quality (fully vendored) | Meta Audiobox-Aesthetics, via OmniNFT reward series |
| `av_align` | sync | Algorithmic AV peak-IoU onset alignment | AV-Align (Yariv et al.), via OmniNFT reward series |
| `av_desync` | sync | Synchformer AV desync, scored as `1/(1+d)` | Synchformer, via OmniNFT reward series |
| `imagebind` | sync | ImageBind multimodal similarity | Meta ImageBind, via OmniNFT reward series |

- **Routing.** `video` rewards drive the video branch, `audio` the audio branch, `sync` both. `audio`/`sync` rewards require `--ltx2_mode av`.
- **Composition.** Combine rewards with weights — GRPO normalises each within its group, so different raw scales mix fine. Example: `--reward_fn "hpsv3:1.0,videoreward:0.5"`. When a low-level reward (e.g. `iqa_quality`) can be satisfied by adding high-frequency noise, pair it with the `anti_noise` guardrail and a learned visual-quality reward; over-weighting `anti_noise` prefers smooth/flat video.
- **Physics.** Use the learned `videoscore2 dims=pc` head for semantic physical commonsense (gravity, collisions, object permanence). A frame statistic cannot tell that a ball should fall down.
- **Differentiable rewards (for `--rl_loss refl`).** A reward declares `kind="differentiable"` and implements `score_grad(samples) -> (Tensor[N], info)`, returning one grad-carrying higher-is-better value per sample. Built-ins are `latent_energy`, `pixel_sharpness` (requires `--vae`), and AV-only `audio_energy` (requires `--refl_av`). Black-box rewards are rejected during ReFL setup.

Reward model code is vendored (`ltx2_rewards/vendor/`), but the runtime libraries of the model-backed rewards are **not** installed with this package — each reward file lists its own `pip install` line at the top of its docstring (`iqa_quality` → `pyiqa`; `hpsv3`/`videoreward`/`videoscore2` → `qwen-vl-utils` (+`peft` for `videoreward`); `imagebind` → `ftfy regex`; `av_align`/`imagebind` have optional exact-parity extras). Checkpoints come from [`huggingface.co/zghhui/OmniNFT-Reward-Series`](https://huggingface.co/zghhui/OmniNFT-Reward-Series) where applicable; pull only the rewards your `--reward_fn` uses.

### Tips
<sub>[↑ contents](#table-of-contents)</sub>

- **Prompts / K.** 8–16 diverse prompts at `K = --rl_group_size` 8 is a working scale. Small `K` gets more groups excluded for zero variance; more diverse prompts reduce those exclusions.
- **Learning rate.** Start at `2e-4` with rank 16. Collapse shows as a falling held-out reward (over-optimization) — lower the LR if seen.
- **Gains are gradual**, bounded by what the policy already sometimes produces. A weak or saturated reward, or too few rounds/prompts, may not move it.
- **Judge by held-out when possible, not by the train reward.** For serious runs, score a fixed-seed **held-out** prompt set between rounds — the train-side reward can climb while the policy degenerates.
- **Keep the peak round, not the last one.** Learned rewards can climb for a stretch and then collapse — every round's LoRA is saved, so pick the one at the held-out peak.
- **Check for hacking.** A composite can be gamed. Pair a low-level objective with a guardrail (e.g. `anti_noise`) and a learned visual-quality reward, score the final RL-LoRA on rewards it was **not** trained on, and look at the samples.
- **Watch:** record `ROUND_REWARD`, the objective-specific `policy` term, and the historically named `kl` scalar. The latter is reference-prediction MSE, not KL divergence. Trends are diagnostics, not guaranteed acceptance criteria; compare held-out scores and generated samples.

### Writing a custom reward
<sub>[↑ contents](#table-of-contents)</sub>

Custom rewards are the reward-design part of RL-LoRA. The reward becomes the language you use to describe the model behavior you want: a detail reward can pull toward richer frames, an anti-noise reward can keep that detail from turning into speckle, a preference model can anchor overall visual quality, and a motion reward can discourage frozen outputs. You are not limited to one score — `--reward_fn` lets you compose several rewards with weights. The creative, testable work is finding a combination whose optimum matches the behavior you actually want.

Any metric you can compute per sample can be a reward, and it does not have to be differentiable, because the update never backpropagates through `score()`: GRPO only compares the `K` scores within a group. But the policy maximizes the number you wrote, not the intent behind it (*reward hacking* — e.g. a consistency metric alone is maximized by a frozen frame); check a trained LoRA against rewards it was **not** trained on. And a metric that scores a group's samples identically (binary or rarely-triggering checks) produces zero-variance groups, which are excluded — no learning signal; prefer dense scores over pass/fail thresholds.

A reward is a small plugin satisfying the `Reward` interface (`ltx2_rewards/registry.py`). It exposes:

- **`score(samples) -> ([float per sample], info)`** — must be higher-is-better (apply any inversion such as `1/(1+distance)` here);
- a **`route`** (`"video"` | `"audio"` | `"sync"`) — which branch the advantage feeds;
- a **`needs`** frozenset of per-sample keys the generator must provide (declaring `"video"` or `"audio_waveform"` makes the generator decode that media for scoring);
- optional `setup(device, **reward_args)` / `teardown()` to load and free a model (called once each, never co-resident with another reward or the DiT).

Register it with the `@register_reward("<name>")` decorator, then select it with `--reward_fn "<name>:<weight>"`. Two ways to make the file importable:

- **Drop-in (no repo edit):** keep the `.py` file anywhere and pass `--reward_plugins path/to/my_reward.py` to Phase A — it is imported before `--reward_fn` is parsed, so the name is usable immediately. With `ltx2_rl_rounds.py`, pass the same flag to the round driver; it forwards the plugin to Phase A and held-out scoring. If you train from an already-created offline rollout cache, the plugin must have been used when that cache was generated, because Phase B replays the stored rewards. Import from the installed package: `from musubi_tuner.ltx2_rewards import BaseReward, register_reward`.
- **Built-in:** place the file in `ltx2_rewards/zoo/` and add one import line in `ltx2_rewards/zoo/__init__.py`.

GRPO advantages, the rollout cache, the loss, VRAM-sequenced setup/teardown, and routing all happen automatically. The `saturation.py` plugin below is a calibration example: simple, dense, and visually verifiable. It rewards more vivid color, so it is not a general quality objective, but it confirms that custom reward loading, scoring, GRPO grouping, and LoRA updates are wired correctly.

```python
from __future__ import annotations

from typing import List, Tuple

from musubi_tuner.ltx2_rewards import BaseReward, register_reward


def _mean_saturation(video, num_frames: int = 5) -> float:
    import torch

    # Decoded videos are usually [C,T,H,W]; some callers keep a leading batch dim.
    t = video[0] if video.dim() == 5 else video
    if t.dim() != 4:
        raise ValueError(f"saturation: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    # Reward code does not need gradients. Move to CPU to keep GPU memory for generation/training.
    t = t.detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    c, frames, _h, _w = t.shape
    if c < 3:
        return 0.0

    rgb = t[:3]
    # Score a small uniform frame sample; dense enough for a demo, cheap enough for rollouts.
    frame_count = min(int(num_frames), frames)
    idx = [round(i * (frames - 1) / max(1, frame_count - 1)) for i in range(frame_count)]
    values = []
    for f in idx:
        frame = rgb[:, f]
        # HSV saturation for RGB can be computed from per-pixel channel max/min.
        cmax = frame.max(dim=0).values
        cmin = frame.min(dim=0).values
        values.append(float(((cmax - cmin) / (cmax + 1e-6)).mean()))
    return sum(values) / len(values) if values else 0.0


@register_reward("saturation")
class SaturationReward(BaseReward):
    kind = "blackbox"
    # route/needs tell the rollout code to decode video and route advantages to the video branch.
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._num_frames = 5

    def setup(self, device, *, num_frames=5, **_ignored) -> None:
        # Values passed through --reward_args are delivered here once before scoring.
        self._num_frames = int(num_frames)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        # Return exactly one higher-is-better scalar per generated sample.
        scores = []
        for sample in samples:
            video = sample.get("video")
            scores.append(_mean_saturation(video, self._num_frames) if video is not None else 0.0)
        return scores, {"reward": "saturation", "num_frames": self._num_frames}
```

Use it as a drop-in file:

```bash
python src/musubi_tuner/ltx2_rl_rounds.py ... ^
  --reward_plugins path/to/saturation.py ^
  --reward_fn "saturation:1.0" ^
  --reward_args num_frames=5
```

The NFT update and several of the reward checkpoints derive from [OmniNFT](https://github.com/zghhui/OmniNFT), a DiffusionNFT-based AV RL project; algorithm references are listed under [References](#references).

---

## Appendix: Full-Parameter Fine-Tuning
<sub>[↑ contents](#table-of-contents)</sub>

> [!NOTE]
> **Scope.** Single-GPU full-rank fine-tuning of large video models is a lightly-documented area. Several techniques here are adapted from the broader LLM optimization literature; measure on your own hardware before committing to long runs.

Full-parameter fine-tuning updates LTX-2 transformer checkpoint weights directly, without attaching or saving a LoRA adapter. Large video transformer training must hold base weights, gradients, optimizer state, activations, and checkpoint-save buffers rather than only a small adapter, so dense full-parameter setups require substantially more GPU memory than LoRA training.

The practical difference from LoRA is the trainable surface. LoRA trains additional low-rank adapter matrices and keeps the base transformer weights frozen; full-parameter fine-tuning changes the transformer weights themselves. This makes the saved artifact a fine-tuned base checkpoint instead of an adapter that must be loaded next to the original checkpoint. It can update weights outside the layers selected by a LoRA target preset. The cost is substantially higher VRAM, larger checkpoints, slower iteration, and greater risk of degrading behavior outside the training distribution if the dataset, learning rate, or regularization are poor.

The same LTX-2 modality paths used by LoRA training are available in full-parameter training. Set `--ltx2_mode video`, `--ltx2_mode av`, or `--ltx2_mode audio` to match the cached dataset. Use `--ltx2_audio_only_model` only when loading a physically audio-only checkpoint variant; it still requires `--ltx2_mode audio`.

Reference-conditioned runs use `--ic_lora_strategy`: `v2v` for video-to-video/reference-video training in video mode, `audio_ref_ic` for reference-audio conditioning in AV or audio mode, `av_ic` for combined video+audio reference conditioning in AV mode, and `video_ref_only_av` for reference-video conditioning with target AV loss in AV mode. These strategies are implemented in `ltx2_train.py` and route the corresponding conditioning and loss path. For full-parameter commands, set `--ic_lora_strategy` explicitly; `--lora_target_preset` is not used to choose full-parameter trainable layers.

Use `ltx2_train.py` for these commands. `ltx2_train_network.py` is the LoRA/network trainer entry point.

### Dataset Preparation
<sub>[↑ contents](#table-of-contents)</sub>

Dataset preparation is the same as LoRA training: create the dataset TOML, cache latents, cache text encoder outputs, and optionally save a dataset manifest. See [Caching Latents](#caching-latents), [Caching Text Encoder Outputs](#caching-text-encoder-outputs), and [Optional: Source-Free Training from Cache](#optional-source-free-training-from-cache).

### Dense bf16 (Adafactor)
<sub>[↑ contents](#table-of-contents)</sub>

Adafactor is the dense bf16 full-parameter optimizer path documented here. It reduces optimizer-state memory by storing factored row and column second-moment statistics for matrix-shaped parameters, rather than full Adam-style moment tensors. The LTX-2.3 Adafactor path uses `--fused_backward_pass`, which adds a per-parameter `step_param` path related to the [optimizer-step-in-backward pattern](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html). When `--max_grad_norm 0` is used, the trainer can step and clear gradients from backward hooks. When global gradient clipping is enabled, as in the benchmark table below, the trainer delays per-parameter stepping until the gradient synchronization point to preserve clipping correctness. The fused Adafactor implementation applies stochastic rounding when fp32 updates are written back into bf16 parameters.

`--adafactor_triton` is an optional fast path for supported contiguous 2D BF16
updates. It avoids parameter-sized fp32 temporaries, requires
`--fused_backward_pass` and the manual-LR configuration shown below
(`scale_parameter=False relative_step=False`), and keeps unsupported
parameters on the regular fused Adafactor path. The active environment must
provide Triton; it is imported only when this flag is enabled.

**Technical tradeoffs**. This is dense bf16 training. The base model weights stay on GPU, every trainable tensor receives an update every optimizer step, and AV/reference modes add activation and conditioning memory. In the measured LTX-2.3 benchmark table, video-only rows are around 54-66 GB; AV and v2v-AV rows can approach or exceed an 80 GB GPU.

Example LTX-2.3 Adafactor command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --learning_rate 1e-5 \
  --optimizer_type Adafactor \
  --optimizer_args scale_parameter=False relative_step=False warmup_init=False weight_decay=0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --adafactor_triton \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_adafactor \
  --output_name ltx23_adafactor_full
```


Add `--save_merged_checkpoint` if you need to write a full merged LTX-2 checkpoint instead of only the trained transformer weights. This can add save-time memory pressure, and repeated merged saves may show slow memory growth, so validate saving separately when running close to the VRAM limit.

`--async_checkpoint_save` takes an immutable CPU snapshot of each periodic transformer checkpoint and writes it in the background while training continues. Use it when checkpoint pauses matter and system RAM can hold one additional checkpoint-sized copy. The next periodic save waits for any previous write, and the final checkpoint remains synchronous. This option is not supported with FSDP, remote-stage training, EMA, Hub uploads, merged checkpoints, streaming Q-GaLore saves, or int8-weight training.

TREAD can also be enabled as a training-time token-routing option. In measured LTX-2.3 runs, `selection_ratio=0.5` reduced step time by about 15-21% depending on mode. Evaluate output quality and convergence separately before using it for long runs because TREAD changes the effective token route. (Credit: [Ada123-a](https://github.com/Ada123-a) — [PR #80](https://github.com/AkaneTendo25/musubi-tuner/pull/80))

```bash
--tread --tread_args target=video selection_ratio=0.5
```

For audio-video runs, use `target=both` if both token streams should be routed:

```bash
--tread --tread_args target=both selection_ratio=0.5
```

The benchmark matrices below are 100-step single-GPU capacity/speed runs on cached video benchmark subsets with dataset `batch_size = 1` and no gradient accumulation. Unless stated otherwise, the tables use bf16 weights/training, FlashAttention, gradient checkpointing, and `blocks_to_swap=0`; runs were checked for finite loss, and resulting checkpoints were separately checked with non-noise previews where sampling was run.

Each row reports **peak VRAM** as the peak resident process GPU memory during the run — the allocator high-water mark, i.e. what must actually fit on the card — and **speed** as the median wall-clock seconds per optimizer step.

> [!NOTE]
> Each benchmark row in this and later benchmark tables was measured on one NVIDIA H100 80GB HBM3 GPU unless explicitly labeled otherwise.

Here is the measured Adafactor full-parameter benchmark matrix on a cached 32-clip video dataset. It used Adafactor with manual learning rate, fused Adafactor stepping, and no block swapping. Each row is `peak VRAM / median s per optimizer step`; `OOM (>80 GB)` means the configuration OOMed.

| Resolution | Frames | video | av | v2v | v2v_av |
| --- | ---: | ---: | ---: | ---: | ---: |
| 832x480 | 17 | 54.2 GB / 1.96 s | 75.6 GB / 3.55 s | 55.6 GB / 2.38 s | 76.4 GB / 4.45 s |
| 832x480 | 33 | 55.7 GB / 2.48 s | 76.8 GB / 4.08 s | 59.2 GB / 3.73 s | 77.8 GB / 5.30 s |
| 832x480 | 49 | 57.4 GB / 3.10 s | 77.0 GB / 4.75 s | 58.7 GB / 4.07 s | 78.7 GB / 6.26 s |
| 832x480 | 65 | 57.7 GB / 3.61 s | 77.4 GB / 5.23 s | 59.5 GB / 5.01 s | 79.2 GB / 7.52 s |
| 832x480 | 81 | 56.0 GB / 4.20 s | 77.4 GB / 5.86 s | 57.0 GB / 5.95 s | OOM (>80 GB) |
| 832x480 | 97 | 57.9 GB / 4.82 s | 79.2 GB / 6.59 s | 58.6 GB / 7.06 s | OOM (>80 GB) |
| 832x480 | 113 | 59.0 GB / 5.44 s | 79.2 GB / 7.28 s | 60.6 GB / 8.11 s | OOM (>80 GB) |
| 832x480 | 129 | 58.9 GB / 6.10 s | 79.0 GB / 7.97 s | 59.9 GB / 9.35 s | OOM (>80 GB) |
| 832x480 | 145 | 60.4 GB / 6.69 s | 79.2 GB / 8.81 s | 61.6 GB / 10.43 s | OOM (>80 GB) |
| 832x480 | 161 | 62.1 GB / 7.37 s | OOM (>80 GB) | 64.4 GB / 11.62 s | OOM (>80 GB) |
| 832x480 | 193 | 58.3 GB / 8.78 s | OOM (>80 GB) | 62.5 GB / 14.38 s | OOM (>80 GB) |
| 832x480 | 241 | 62.7 GB / 10.96 s | OOM (>80 GB) | 65.4 GB / 18.83 s | OOM (>80 GB) |
| 1280x720 | 17 | 57.4 GB / 3.05 s | 76.9 GB / 4.61 s | 58.7 GB / 4.68 s | 78.8 GB / 7.80 s |
| 1280x720 | 33 | 56.9 GB / 4.42 s | 78.5 GB / 6.07 s | 58.6 GB / 6.37 s | OOM (>80 GB) |
| 1280x720 | 49 | 58.4 GB / 5.84 s | 78.2 GB / 7.66 s | 59.2 GB / 8.94 s | OOM (>80 GB) |
| 1280x720 | 65 | 62.3 GB / 7.33 s | OOM (>80 GB) | 64.2 GB / 11.70 s | OOM (>80 GB) |
| 1280x720 | 81 | 58.6 GB / 8.91 s | OOM (>80 GB) | 63.6 GB / 14.64 s | OOM (>80 GB) |
| 1280x720 | 97 | 62.7 GB / 10.55 s | OOM (>80 GB) | 65.0 GB / 17.90 s | OOM (>80 GB) |
| 1280x720 | 113 | 61.1 GB / 12.25 s | OOM (>80 GB) | 66.0 GB / 21.35 s | OOM (>80 GB) |
| 1280x720 | 129 | 62.2 GB / 14.00 s | OOM (>80 GB) | 68.5 GB / 25.23 s | OOM (>80 GB) |
| 1280x720 | 145 | 60.2 GB / 15.82 s | OOM (>80 GB) | 69.2 GB / 29.45 s | OOM (>80 GB) |
| 1280x720 | 161 | 62.3 GB / 17.73 s | OOM (>80 GB) | 72.2 GB / 33.85 s | OOM (>80 GB) |
| 1280x720 | 193 | 64.1 GB / 21.56 s | OOM (>80 GB) | 75.5 GB / 42.50 s | OOM (>80 GB) |
| 1280x720 | 241 | 66.5 GB / 27.96 s | OOM (>80 GB) | OOM (>80 GB) | OOM (>80 GB) |

> [!WARNING]
> The example command above uses `--max_grad_norm 1.0` (gradient clipping on). With the fused backward pass, clipping requires all gradients to be resident at once to compute the global norm; setting `--max_grad_norm 0` disables it and lets each gradient be freed as its parameter is stepped, lowering peak VRAM by roughly one copy of the trainable gradients (about 2 bytes per parameter in bf16, on the order of 25 GB for this model). Gradient clipping is a standard safeguard against exploding gradients, so keep `--max_grad_norm 1.0` unless VRAM requires disabling it.

> [!NOTE]
> Full fine-tuning supports standard bidirectional block swap. See [Block Swap for Full Fine-Tuning](#block-swap-for-full-fine-tuning) for the measured 24 GB profile.

#### Using Full-Finetune Checkpoints for Inference

`ltx2_train.py` saves a transformer-only checkpoint with internal key naming (`model.<...>`) and without the `config` safetensors metadata. The LTX-2 loaders (generation, caching, LoRA training) expect a self-contained bundle in the base-checkpoint layout (`model.diffusion_model.<...>` plus VAE / vocoder / audio VAE / text-projection weights and the `config` metadata), so the trainer output cannot be passed to `--ltx2_checkpoint` directly. Merge it into a copy of the base bundle first:

```bash
python ltx2_merge_fft_to_model.py \
  --fft_checkpoint output/full_ft_adafactor/ltx23_adafactor_full.safetensors \
  --base_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --output output/full_ft_adafactor/ltx23_adafactor_full_bundle.safetensors
```

The merged bundle loads anywhere the base checkpoint does. The utility fails loudly if any trained tensor cannot be matched to the bundle, so a successful merge guarantees nothing was silently dropped.

### BAdam Block-Coordinate
<sub>[↑ contents](#table-of-contents)</sub>

BAdam applies block-coordinate optimization to full-parameter training. The [BAdam paper](https://arxiv.org/abs/2404.02827) presents this as a memory-efficient way to combine block coordinate descent with Adam-style updates. In the LTX-2.3 implementation, the transformer parameters are split into block groups. At each step, one block group and selected non-block parameters are trainable; updates are written directly into the base checkpoint weights.

Every training step still runs the full transformer forward through all blocks and computes the normal diffusion loss. BAdam changes the backward/update side: only the currently active block group has trainable block parameters and optimizer state, while inactive block weights stay frozen for that window.

Inactive block parameters are frozen with `requires_grad=False`, and their optimizer state can be purged at block switches. The frozen blocks remain part of the transformer computation. For a step with active block `k`, the full model runs forward; later operations are still needed by autograd so gradients can reach block `k`, while frozen parameters do not receive parameter gradients. As a result, BAdam `s/step` values are full-model training-step timings for the active-block schedule.

With `switch_block_every=25` on the measured 48-block LTX-2.3 transformer, the optimizer trains one block group for 25 steps, then advances to the next group. With `block_group_size=1`, a 100-step benchmark covers four sequential block windows and one pass over the 48 block groups takes 1,200 steps. This schedule lowers active gradient and optimizer-state memory. It also means a given block is updated less often than in dense Adafactor, where every trainable tensor receives an update every optimizer step.

`block_group_size` controls how many consecutive inferred transformer blocks are placed into one active BAdam group. `block_group_size=1` gives the lowest active gradient/optimizer-state memory. Larger values such as `2`, `4`, or `8` train more blocks at once, use more VRAM, reduce the number of block windows per full sweep, and move the run closer to dense full-parameter fine-tuning. This is the main BAdam quality/speed/VRAM tradeoff: if full dense fine-tuning fits, BAdam is usually unnecessary; if dense training does not fit, increase `block_group_size` up to the VRAM budget.

When `use_fp32_active_copy=True` (the recommended memory-safe path), keep `reset_state_on_switch=True`. The base optimizer state is attached to temporary fp32 copies of the active block parameters; this implementation does not remap Adam state from one active window to the next.

**Technical tradeoffs**. BAdam reduces active optimizer and gradient state by updating only the active block window. It does not remove the dense base weights, and it still runs the full transformer computation every step. A full block sweep takes many optimizer steps, so learning is distributed over the model more slowly than dense Adafactor. Because each block updates less often, convergence can be slower and BAdam may reach a higher loss plateau than dense Adafactor on this model. For LTX-2.3 FFT, use BAdam as a lower-VRAM fallback: prefer dense Adafactor when it fits, or Q-GaLore/QAPOLLO/int8-weight training for 24 GB-class video-only setups. The [BREAD paper](https://openreview.net/pdf?id=zs6bRl05g8) analyzes suboptimal landscapes in block-coordinate LLM fine-tuning and proposes inactive-block correction as a mitigation.

BREAD-SGD is an optional block-coordinate correction, implemented through post-accumulate gradient hooks. It applies a correction to inactive-block parameters to mitigate the block-coordinate plateau; evaluate convergence and quality on your own data before relying on it. `bread_sgd_mode=all` applies the correction to all inactive parameters and has the highest extra memory cost. `bread_sgd_mode=partial` applies it only to downstream block groups that already sit on the backward path after the active block. `bread_sgd_mode=window` bounds that correction to `bread_sgd_window_blocks` downstream block groups. The `partial` and `window` modes reduce extra weight-gradient pressure compared with `all`.

References: [BAdam paper](https://arxiv.org/abs/2404.02827), [BREAD paper](https://openreview.net/forum?id=zs6bRl05g8), [BlockOptimizers implementation](https://github.com/microsoft/BlockOptimizers).

The LTX-2.3 BAdam command below uses bitsandbytes `AdamW8bit` as the base optimizer and gradient release enabled. Keep `--base_optimizer_args` matched to the selected base optimizer.

Example LTX-2.3 BAdam command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --learning_rate 1e-5 \
  --optimizer_type BAdam \
  --optimizer_args base_optimizer_type=AdamW8bit switch_block_every=25 switch_mode=ascending block_group_size=2 include_non_block=True use_fp32_active_copy=True purge_inactive_state=True reset_state_on_switch=True use_gradient_release=True \
  --max_grad_norm 1.0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_badam \
  --output_name ltx23_badam_full
```

Memory-bounded BREAD-SGD ablation, for convergence experiments only:

```bash
--optimizer_args base_optimizer_type=AdamW8bit switch_block_every=25 switch_mode=descending include_non_block=True use_fp32_active_copy=True purge_inactive_state=True reset_state_on_switch=True use_gradient_release=True bread_sgd=True bread_sgd_mode=window bread_sgd_window_blocks=1 bread_sgd_lr_scale=5.0
```

Measured BAdam full-parameter benchmark matrix on the exact-resolution cached video subset used for the BAdam/Q-GaLore/QAPOLLO sweeps. It used bf16 model weights, gradient checkpointing, FlashAttention, BAdam with bitsandbytes AdamW8bit, `switch_block_every=25`, `block_group_size=1`, `include_non_block=True`, `use_fp32_active_copy=True`, `purge_inactive_state=True`, `reset_state_on_switch=True`, `use_gradient_release=True`, and no block swapping.
Each row is `peak VRAM / median s per optimizer step`; `OOM (>80 GB)` means the configuration OOMed. These timings do not measure full-model convergence rate because BAdam updates only the active block window.

| Resolution | Frames | video | av | v2v | v2v_av |
| --- | ---: | ---: | ---: | ---: | ---: |
| 832x480 | 17 | 43.6 GB / 1.41 s | 44.1 GB / 3.16 s | 43.5 GB / 1.90 s | 44.1 GB / 3.41 s |
| 832x480 | 33 | 43.2 GB / 2.07 s | 44.1 GB / 3.60 s | 43.6 GB / 2.68 s | 45.4 GB / 3.70 s |
| 832x480 | 49 | 42.4 GB / 2.58 s | 44.1 GB / 4.04 s | 42.0 GB / 3.23 s | 47.0 GB / 4.76 s |
| 832x480 | 65 | 42.0 GB / 3.17 s | 44.9 GB / 4.84 s | 43.6 GB / 4.14 s | 48.6 GB / 6.36 s |
| 832x480 | 81 | 43.4 GB / 3.53 s | 45.7 GB / 5.57 s | 43.6 GB / 5.14 s | 50.3 GB / 7.38 s |
| 832x480 | 97 | 43.3 GB / 4.35 s | 46.6 GB / 6.26 s | 43.0 GB / 6.16 s | 52.0 GB / 8.85 s |
| 832x480 | 113 | 42.8 GB / 4.77 s | 47.5 GB / 6.39 s | 43.6 GB / 7.47 s | 79.2 GB / 9.39 s |
| 832x480 | 129 | 42.8 GB / 5.50 s | 48.3 GB / 7.77 s | 42.0 GB / 8.35 s | 55.4 GB / 11.20 s |
| 832x480 | 145 | 43.6 GB / 6.22 s | 49.1 GB / 8.58 s | 43.5 GB / 9.54 s | 57.2 GB / 12.18 s |
| 832x480 | 161 | 42.8 GB / 6.90 s | 50.0 GB / 8.97 s | 43.6 GB / 10.61 s | 58.9 GB / 13.93 s |
| 832x480 | 193 | 40.6 GB / 8.02 s | 51.6 GB / 10.10 s | 45.6 GB / 13.46 s | 62.3 GB / 16.34 s |
| 832x480 | 241 | 43.3 GB / 10.08 s | 54.4 GB / 12.35 s | 49.7 GB / 17.43 s | 67.1 GB / 20.83 s |
| 1280x720 | 17 | 43.4 GB / 2.60 s | 44.1 GB / 3.60 s | 43.6 GB / 3.40 s | 46.7 GB / 5.05 s |
| 1280x720 | 33 | 43.6 GB / 3.82 s | 45.8 GB / 5.55 s | 43.6 GB / 5.32 s | 50.6 GB / 7.27 s |
| 1280x720 | 49 | 43.0 GB / 5.27 s | 47.7 GB / 6.50 s | 42.0 GB / 7.89 s | 54.3 GB / 10.44 s |
| 1280x720 | 65 | 42.0 GB / 6.61 s | 49.6 GB / 8.75 s | 42.6 GB / 10.26 s | 58.3 GB / 13.70 s |
| 1280x720 | 81 | 42.0 GB / 7.88 s | 51.5 GB / 9.62 s | 45.5 GB / 13.34 s | 62.0 GB / 16.85 s |
| 1280x720 | 97 | 43.2 GB / 9.34 s | 53.5 GB / 11.36 s | 48.6 GB / 16.20 s | 65.9 GB / 19.55 s |
| 1280x720 | 113 | 42.6 GB / 11.22 s | 55.4 GB / 13.23 s | 51.2 GB / 19.82 s | 69.5 GB / 23.13 s |
| 1280x720 | 129 | 42.0 GB / 12.86 s | 57.2 GB / 15.04 s | 54.1 GB / 23.07 s | 73.5 GB / 27.12 s |
| 1280x720 | 145 | 43.3 GB / 14.45 s | 59.1 GB / 17.34 s | 56.8 GB / 26.98 s | 77.2 GB / 31.36 s |
| 1280x720 | 161 | 44.9 GB / 16.45 s | 61.1 GB / 19.38 s | 59.7 GB / 31.11 s | OOM (>80 GB) |
| 1280x720 | 193 | 47.9 GB / 20.04 s | 65.1 GB / 23.08 s | 66.0 GB / 40.00 s | OOM (>80 GB) |
| 1280x720 | 241 | 52.1 GB / 26.10 s | 70.5 GB / 29.76 s | 74.2 GB / 54.73 s | OOM (>80 GB) |

BAdam speed numbers should be read together with the active-block schedule. They are real wall-clock step timings, while the optimizer update is block-local for the active window.

If the base optimizer supports its own stochastic rounding or optimizer-specific update modes, pass those through `--base_optimizer_args`. Keep those arguments matched to the selected base optimizer.

### Q-GaLore Quantized
<sub>[↑ contents](#table-of-contents)</sub>

Q-GaLore is a quantized full-parameter fine-tuning path. Eligible LTX-2 `Linear` modules are replaced with `QGaLoreLinear` wrappers that store 8-bit quantized base weights. During training, wrapped weights are dequantized for forward/backward, dense gradients are computed for those weights, gradients are projected to rank `--qgalore_rank`, and bitsandbytes AdamW8bit applies the projected update back to the base weight shape. The updated weights are quantized again for runtime storage. With `--qgalore_dequantize_save`, enabled by default, checkpoints are saved as standard checkpoint tensors. `--qgalore_streaming_dequantize_save` is an optional lower-VRAM checkpoint export path that dequantizes and writes one selected `Linear` weight at a time; its default temporary dequantization device is CPU.

**Technical tradeoffs**. Rank, projection refresh interval, projection scale, projection quantization, and target selection are training hyperparameters. Lower rank reduces memory and constrains the projected update. The `QGaLoreLinear` wrapper stores selected base weights in 8-bit form; `--qgalore_weight_group_size 0` uses row-wise scales for those weights (one scale per output channel) and is the recommended default; values greater than 0 use flattened groups. Projection matrices can be quantized separately with `--qgalore_proj_quant`, `--qgalore_proj_bits`, and `--qgalore_proj_group_size`. In the measured tables this is a VRAM reduction path, not a speed path: wrapped weights are dequantized for forward/backward, dense gradients are computed, and projection/update/requantization work is added around the optimizer step. Narrower target sets reduce the number of Q-GaLore-wrapped weights. Unwrapped trainable parameters remain in standard optimizer groups unless another freeze or learning-rate-scale option disables them. `--qgalore_load_device cpu` lowers the initial GPU peak by loading, replacing, and quantizing the transformer on CPU before moving the quantized modules to GPU. `--qgalore_targets video` Q-GaLore-wraps eligible Linear weights in the video stream inside each transformer block: `attn1`, `attn2`, and `ff`. It does not Q-GaLore-wrap audio/cross-audio modules or non-block Linear layers. `--qgalore_targets all` wraps every eligible Linear layer, including non-block layers and audio-side layers when they are present.

Required constraints are enforced by the trainer:

- `--qgalore_full_ft` requires `--optimizer_type QGaLoreAdamW8bit` or `--optimizer_type QAPOLLOAdamW` unless the optimizer type is omitted, in which case the trainer defaults to `QGaLoreAdamW8bit`.
- `--qgalore_full_ft` requires `--fused_backward_pass`.
- Q-GaLore fused backward requires `--max_grad_norm 0`.
- `--qgalore_full_ft` cannot be combined with `--fp8_base`, `--fp8_scaled`, or `--nf4_base`. Those are generic base-weight quantization paths; Q-GaLore uses its own trainable quantized `Linear` wrappers for the selected weights.

Example LTX-2.3 Q-GaLore all-target command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --learning_rate 1e-5 \
  --optimizer_type QGaLoreAdamW8bit \
  --optimizer_args weight_decay=0.0 min_8bit_size=4096 \
  --qgalore_full_ft \
  --qgalore_load_device cpu \
  --qgalore_targets all \
  --qgalore_rank 256 \
  --qgalore_update_proj_gap 200 \
  --qgalore_scale 0.25 \
  --qgalore_proj_quant \
  --qgalore_proj_bits 4 \
  --qgalore_proj_group_size 256 \
  --qgalore_svd_method lowrank \
  --qgalore_svd_oversampling 32 \
  --qgalore_svd_niter 1 \
  --qgalore_weight_bits 8 \
  --qgalore_weight_group_size 0 \
  --qgalore_stochastic_round \
  --max_grad_norm 0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_qgalore \
  --output_name ltx23_qgalore_full
```

Measured Q-GaLore all-target benchmark matrix on the exact-resolution cached video subset used for the BAdam/Q-GaLore/QAPOLLO sweeps. It used bf16 compute, gradient checkpointing, FlashAttention, Q-GaLore AdamW8bit, `rank=256`, `targets=all`, low-rank SVD, 4-bit projection quantization, stochastic rounding, fused backward, `--max_grad_norm 0`, `--qgalore_load_device cpu`, and no block swapping.
Each row is `peak VRAM / median s per optimizer step`; `OOM (>80 GB)` means the configuration OOMed.

| Resolution | Frames | video | av | v2v | v2v_av |
| --- | ---: | ---: | ---: | ---: | ---: |
| 832x480 | 17 | 23.5 GB / 2.74 s | 32.0 GB / 7.02 s | 25.1 GB / 3.66 s | 33.6 GB / 7.17 s |
| 832x480 | 33 | 24.3 GB / 3.91 s | 32.9 GB / 7.46 s | 25.8 GB / 4.54 s | 34.5 GB / 7.50 s |
| 832x480 | 49 | 24.8 GB / 4.13 s | 33.7 GB / 8.18 s | 27.2 GB / 5.36 s | 37.6 GB / 9.37 s |
| 832x480 | 65 | 25.4 GB / 4.59 s | 34.4 GB / 8.71 s | 28.6 GB / 6.28 s | 38.5 GB / 9.75 s |
| 832x480 | 81 | 26.4 GB / 5.72 s | 35.5 GB / 7.88 s | 30.2 GB / 7.42 s | 40.3 GB / 11.11 s |
| 832x480 | 97 | 27.3 GB / 6.07 s | 36.6 GB / 10.45 s | 30.6 GB / 8.50 s | 41.4 GB / 12.39 s |
| 832x480 | 113 | 27.9 GB / 6.87 s | 37.7 GB / 10.38 s | 32.7 GB / 9.33 s | 44.1 GB / 13.39 s |
| 832x480 | 129 | 28.8 GB / 7.56 s | 38.0 GB / 11.69 s | 33.8 GB / 10.18 s | 45.8 GB / 14.80 s |
| 832x480 | 145 | 28.9 GB / 8.04 s | 38.8 GB / 11.19 s | 35.2 GB / 11.96 s | 46.7 GB / 15.56 s |
| 832x480 | 161 | 29.2 GB / 8.60 s | 39.4 GB / 11.45 s | 35.8 GB / 12.73 s | 48.1 GB / 16.96 s |
| 832x480 | 193 | 30.4 GB / 9.91 s | 41.4 GB / 13.39 s | 36.2 GB / 15.42 s | 49.6 GB / 20.05 s |
| 832x480 | 241 | 32.9 GB / 12.01 s | 44.4 GB / 17.79 s | 39.9 GB / 19.59 s | 54.5 GB / 24.31 s |
| 1280x720 | 17 | 24.8 GB / 4.17 s | 33.9 GB / 7.89 s | 27.2 GB / 4.85 s | 37.1 GB / 7.71 s |
| 1280x720 | 33 | 27.1 GB / 5.51 s | 36.2 GB / 8.18 s | 30.8 GB / 7.78 s | 40.5 GB / 10.36 s |
| 1280x720 | 49 | 28.1 GB / 6.62 s | 38.2 GB / 9.66 s | 33.2 GB / 9.56 s | 44.9 GB / 14.11 s |
| 1280x720 | 65 | 29.4 GB / 8.14 s | 39.2 GB / 12.80 s | 35.7 GB / 12.61 s | 48.1 GB / 17.55 s |
| 1280x720 | 81 | 30.4 GB / 9.73 s | 41.3 GB / 15.01 s | 36.3 GB / 15.84 s | 49.4 GB / 21.39 s |
| 1280x720 | 97 | 40.5 GB / 11.53 s | 43.9 GB / 15.70 s | 38.6 GB / 18.83 s | 53.0 GB / 23.91 s |
| 1280x720 | 113 | 34.4 GB / 13.18 s | 79.1 GB / 17.72 s | 41.8 GB / 22.30 s | 59.1 GB / 27.83 s |
| 1280x720 | 129 | 47.5 GB / 14.99 s | 47.6 GB / 20.46 s | 44.2 GB / 25.68 s | 61.3 GB / 32.49 s |
| 1280x720 | 145 | 36.7 GB / 16.79 s | 49.0 GB / 22.21 s | 46.8 GB / 29.95 s | 65.4 GB / 36.99 s |
| 1280x720 | 161 | 36.3 GB / 18.59 s | 48.5 GB / 24.30 s | 49.9 GB / 33.98 s | 68.6 GB / 40.30 s |
| 1280x720 | 193 | 37.6 GB / 22.51 s | 52.3 GB / 27.43 s | 73.4 GB / 43.20 s | 76.9 GB / 50.16 s |
| 1280x720 | 241 | 42.1 GB / 28.63 s | 58.6 GB / 34.89 s | 66.0 GB / 58.59 s | OOM (>80 GB) |

### APOLLO and QAPOLLO
<sub>[↑ contents](#table-of-contents)</sub>

APOLLO (`apollo-torch`) is an optimizer-state memory reduction method for full-parameter training. `APOLLOAdamW` applies APOLLO low-rank auxiliary optimizer state to 2-D trainable tensors; non-2-D tensors stay in ordinary AdamW-style groups inside the same optimizer. Dense APOLLO keeps the base checkpoint weights as normal dense weights, so it reduces optimizer-state memory without reducing memory used by dense base weights.

`QAPOLLOAdamW` combines APOLLO optimizer updates with the same quantized `Linear` replacement path used by Q-GaLore. Enable it with `--qgalore_full_ft --optimizer_type QAPOLLOAdamW`. The selected `Linear` weights are stored through the `QGaLoreLinear` wrappers, while optimizer projection settings come from the `--apollo_*` arguments. In QAPOLLO runs, `--qgalore_targets`, `--qgalore_load_device`, `--qgalore_weight_bits`, `--qgalore_weight_group_size`, `--qgalore_stochastic_round`, and `--qgalore_dequantize_save` control the quantized wrapper behavior; use `--apollo_rank`, `--apollo_update_proj_gap`, `--apollo_scale`, `--apollo_proj`, `--apollo_proj_type`, and `--apollo_scale_type` for the optimizer projection. The shared `qgalore_*` arguments describe the quantized storage wrapper; the APOLLO update rule is selected by `QAPOLLOAdamW` and the `apollo_*` arguments. QAPOLLO uses `optim_bits=8` by default for bitsandbytes optimizer state; `optim_bits=32` is available as a diagnostic or stability fallback and uses more optimizer-state memory.

**Technical tradeoffs**. The default APOLLO projection is `--apollo_proj random`, so it does not run Q-GaLore SVD initialization or SVD refresh. Dense `APOLLOAdamW` still keeps dense base weights. `QAPOLLOAdamW` has the same dequantize-forward/backward and requantize-storage costs as the quantized Q-GaLore wrapper path, but uses APOLLO's projected-gradient scaling update instead of the Q-GaLore AdamW8bit projected-update rule.

Example LTX-2.3 QAPOLLO all-target command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --learning_rate 1e-5 \
  --optimizer_type QAPOLLOAdamW \
  --optimizer_args weight_decay=0.0 min_8bit_size=4096 optim_bits=8 \
  --qgalore_full_ft \
  --qgalore_targets all \
  --qgalore_weight_bits 8 \
  --qgalore_weight_group_size 0 \
  --qgalore_stochastic_round \
  --apollo_rank 256 \
  --apollo_update_proj_gap 200 \
  --apollo_scale 1.0 \
  --apollo_proj random \
  --apollo_proj_type std \
  --apollo_scale_type channel \
  --max_grad_norm 0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --qgalore_streaming_dequantize_save \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_qapollo \
  --output_name ltx23_qapollo_full
```

Common APOLLO/QAPOLLO configuration variants:

- For dense APOLLO without quantized `Linear` storage, use `--optimizer_type APOLLOAdamW`, remove the `--qgalore_*` arguments, and set `--max_grad_norm 1.0` if global gradient clipping is needed. This keeps the base transformer weights dense.
- For narrower quantized coverage, change `--qgalore_targets all` to `--qgalore_targets video`. This wraps only video-side eligible `Linear` weights and reduces the wrapped parameter set, but does not update the audio-side eligible `Linear` weights through the quantized wrapper path.
- `--apollo_rank`, `--apollo_update_proj_gap`, `--apollo_scale`, `--apollo_proj`, `--apollo_proj_type`, and `--apollo_scale_type` control the APOLLO optimizer projection. Higher rank increases optimizer memory and may change convergence.
- `optim_bits=8` or `optim_bits=32` controls QAPOLLO optimizer-state storage width. It is separate from model-weight precision and from the Q-GaLore wrapper's `--qgalore_weight_bits`.

Measured QAPOLLO all-target rows on the exact-resolution cached video subset used for the BAdam/Q-GaLore/QAPOLLO sweeps. It used bf16 compute, gradient checkpointing, FlashAttention, QAPOLLOAdamW with `optim_bits=8`, `rank=256`, `targets=all`, random APOLLO projection, stochastic rounding, fused backward, `--max_grad_norm 0`, default `--qgalore_load_device cuda`, and no block swapping.
Each row is `training-phase peak VRAM / median s per optimizer step`; `OOM (>80 GB)` means the configuration OOMed. QAPOLLO can have a higher temporary peak while loading and replacing the base transformer. Use `--qgalore_load_device cpu` when the load-time peak is the limiting factor.
The `optim_bits=8` QAPOLLO path was checked for finite loss and non-noise preview generation. Quantized `Linear` full-parameter paths still need longer large-dataset validation; stable convergence settings and numerical-stability margins are not established by this benchmark table.

| Resolution | Frames | video | av | v2v | v2v_av |
| --- | ---: | ---: | ---: | ---: | ---: |
| 832x480 | 17 | 22.2 GB / 2.46 s | 32.2 GB / 4.97 s | 22.5 GB / 2.85 s | 34.5 GB / 5.35 s |
| 832x480 | 33 | 22.6 GB / 3.01 s | 32.4 GB / 5.54 s | 23.2 GB / 3.69 s | 35.6 GB / 6.28 s |
| 832x480 | 49 | 22.5 GB / 3.61 s | 34.3 GB / 6.13 s | 24.5 GB / 4.61 s | 37.6 GB / 7.30 s |
| 832x480 | 65 | 23.1 GB / 4.21 s | 35.9 GB / 6.81 s | 25.9 GB / 5.57 s | 39.6 GB / 8.41 s |
| 832x480 | 81 | 24.4 GB / 4.82 s | 36.7 GB / 7.47 s | 27.6 GB / 6.56 s | 43.2 GB / 9.56 s |
| 832x480 | 97 | 24.6 GB / 5.48 s | 37.6 GB / 8.12 s | 28.9 GB / 7.65 s | 42.9 GB / 10.78 s |
| 832x480 | 113 | 25.1 GB / 6.10 s | 39.3 GB / 8.90 s | 30.7 GB / 8.74 s | 45.3 GB / 12.15 s |
| 832x480 | 129 | 26.4 GB / 6.69 s | 40.2 GB / 9.52 s | 32.9 GB / 9.99 s | 48.3 GB / 13.45 s |
| 832x480 | 145 | 26.4 GB / 7.41 s | 40.2 GB / 10.35 s | 33.7 GB / 11.11 s | 47.9 GB / 14.83 s |
| 832x480 | 161 | 26.6 GB / 8.03 s | 40.2 GB / 11.09 s | 33.4 GB / 12.33 s | 49.5 GB / 16.15 s |
| 832x480 | 193 | 28.4 GB / 9.44 s | 43.8 GB / 12.58 s | 35.3 GB / 15.09 s | 52.5 GB / 19.07 s |
| 832x480 | 241 | 30.9 GB / 11.63 s | 46.8 GB / 15.06 s | 39.4 GB / 19.27 s | 58.2 GB / 23.95 s |
| 1280x720 | 17 | 22.8 GB / 3.54 s | 34.7 GB / 6.04 s | 24.8 GB / 4.53 s | 37.1 GB / 7.47 s |
| 1280x720 | 33 | 24.2 GB / 4.89 s | 36.9 GB / 7.43 s | 27.7 GB / 6.74 s | 43.0 GB / 9.82 s |
| 1280x720 | 49 | 25.5 GB / 6.30 s | 38.9 GB / 9.16 s | 31.8 GB / 9.18 s | 47.2 GB / 12.55 s |
| 1280x720 | 65 | 27.2 GB / 7.81 s | 40.5 GB / 10.72 s | 34.2 GB / 12.05 s | 50.4 GB / 15.63 s |
| 1280x720 | 81 | 28.4 GB / 9.35 s | 43.1 GB / 12.50 s | 35.2 GB / 14.97 s | 52.3 GB / 19.08 s |
| 1280x720 | 97 | 30.6 GB / 10.88 s | 45.0 GB / 14.22 s | 38.4 GB / 18.22 s | 56.4 GB / 22.44 s |
| 1280x720 | 113 | 32.1 GB / 12.73 s | 48.4 GB / 16.12 s | 41.7 GB / 21.64 s | 60.3 GB / 26.20 s |
| 1280x720 | 129 | 33.2 GB / 14.25 s | 48.0 GB / 18.05 s | 44.9 GB / 25.11 s | 64.4 GB / 30.36 s |
| 1280x720 | 145 | 33.9 GB / 16.05 s | 50.0 GB / 19.74 s | 47.5 GB / 29.27 s | 68.4 GB / 34.51 s |
| 1280x720 | 161 | 34.4 GB / 18.07 s | 51.4 GB / 21.88 s | 49.8 GB / 33.23 s | 72.2 GB / 39.40 s |
| 1280x720 | 193 | 38.1 GB / 21.76 s | 55.4 GB / 26.22 s | 57.3 GB / 42.16 s | 79.0 GB / 48.74 s |
| 1280x720 | 241 | 43.3 GB / 28.27 s | 61.8 GB / 32.95 s | 66.2 GB / 57.67 s | OOM (>80 GB) |

### Int8 / Int8 ConvRot / EMA
<sub>[↑ contents](#table-of-contents)</sub>

> [!WARNING]
> **Experimental.** Int8 weight storage is an inherently lossy training path: updates land on an int8 grid through stochastic rounding with no high-precision master copy, so the final weights carry residual rounding noise that shows up as lost high-frequency detail in samples. In the checked runs, pairing the run with an EMA of the dequantized weights (described below) recovered that detail, at the cost of host RAM and save-time disk. Train in bf16 when the hardware allows it.

**Int8 storage.** `--int8_weights` stores trainable `Linear` weights as int8 (symmetric, per-row or per-group scale) with no bf16 master; the optimizer updates the int8 storage in place through stochastic rounding, while forward/backward dequantize to bf16 for the GEMM.

- Requires `--fused_backward_pass`; there is no automatic optimizer default for this path, so pass `--optimizer_type` explicitly. `--fused_backward_pass` accepts Adafactor, CAME/CAME8bit, Lion, APOLLO, BAdam, and others; with `--int8_weights` choose a non-quantized optimizer. Mutually exclusive with `--fp8_gemm`, `--qgalore_full_ft`, `--fp8_scaled`.
- With `--max_grad_norm 0` (gradient clipping off), short-context video-only training fits in roughly 21-23 GB: int8 storage roughly halves the trainable weight bytes, and without the global-norm clip each gradient is freed as its parameter is stepped. `--max_grad_norm 1.0` keeps the full gradient buffer resident and raises peak VRAM.
- Saving a full checkpoint dequantizes the int8 weights back to bf16; pass `--mem_eff_save` to stream that conversion one tensor at a time so the save's peak memory stays within the training envelope (the resulting checkpoint is identical either way).
- Validate save/resume on the target GPU before long runs. The loss trajectory diverges from a bf16 run.

Grid controls: `--int8_weights_targets` (same tokens as `--qgalore_targets`), `--int8_weights_group_size N` (`0` = row-wise scale, `>0` = group-wise along the input dimension; smaller groups give a tighter grid for a small VRAM cost in extra scales), `--int8_weights_outlier_quantile q` (`1.0` = absmax, `<1.0` = clip the scale to a per-row/group quantile of `|w|`), `--int8_weights_sparse_ratio r` (dense-and-sparse, [arXiv:2310.07147](https://arxiv.org/abs/2310.07147): keep the top fraction `r` of `|w|` as an exact fp32 side-vector excluded from the int8 grid; an alternative to `--int8_weights_outlier_quantile`, use one or the other), `--int8_weights_min_numel` to skip small layers.

**Int8 ConvRot.** `--int8_weights_convrot` (`''` = off; `auto` or an explicit group size) rotates the int8 weight grid with a group-wise Hadamard transform at quantization and inverts the rotation at dequantization. The rotation spreads per-group outliers, so the grid is tighter and per-step quantization error is lower; the GEMM still runs on real-space bf16 weights, so gradients and the optimizer are unchanged. `auto` picks the group size per layer.

**Int8 W8A8 compute.** `--int8_weights_w8a8` keeps the `--int8_weights` resident-weight path, but runs the forward and grad-input GEMMs through int8 W8A8 compute when the activation/device support it. Grad-weight stays bf16 for stochastic-rounding update quality. This requires `--int8_weights_group_size 0` and `--int8_weights_sparse_ratio 0`; otherwise the trainer rejects the run.

**EMA compensation.** Stochastic-rounding noise is zero-mean, so an exponential moving average of the dequantized weights averages it out across update events — and because the average is not constrained to the int8 grid, the saved `*_ema.safetensors` checkpoint can express detail that the rounded weights cannot. Enable it with `--use_ema --ema_decay 0.95 --ema_update_after_step 200 --ema_update_every 25 --ema_cpu_offload`, and sample/publish the EMA checkpoint rather than the final weights. Costs to plan for:

- **Host RAM:** with `--ema_cpu_offload` the shadow stores about two bytes per trainable parameter in host RAM (~40 GB for video-target training on the 22B model), on top of the trainer's normal host usage. Without `--ema_cpu_offload` the shadow lives on the GPU and adds the same amount to VRAM, which defeats the low-VRAM point of this path — keep the offload on for VRAM-constrained cards.
- **Wall time:** each EMA update dequantizes the tracked weights and copies them to the shadow's device; at 22B scale this takes seconds per event, so the amortized overhead is set by `--ema_update_every`.
- **Disk:** every save writes an additional full-size `*_ema.safetensors` next to the regular checkpoint.

Example LTX-2.3 int8 ConvRot + EMA command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --int8_weights \
  --int8_weights_convrot auto \
  --int8_weights_targets video \
  --int8_weights_group_size 64 \
  --learning_rate 3e-5 \
  --optimizer_type Adafactor \
  --optimizer_args scale_parameter=False relative_step=False warmup_init=False weight_decay=0.1 \
  --max_grad_norm 0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --use_ema \
  --ema_decay 0.95 \
  --ema_update_after_step 200 \
  --ema_update_every 25 \
  --ema_cpu_offload \
  --mem_eff_save \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_int8cr \
  --output_name ltx23_int8cr_video
```

The learning rate is set modestly higher than the equivalent bf16 Adafactor command: updates far below the int8 grid step are realized only occasionally under stochastic rounding, so a larger step partially offsets that. Sample the `*_ema.safetensors` checkpoint, not the final weights.

This path was validated on short video-only fine-tuning runs: finite loss, save/resume, and sample checks in which the EMA checkpoint restored high-frequency detail that the final stochastically-rounded weights lost. Long-run convergence behavior and numerical-stability margins are not established.

The int8 grid is rebuilt from the bf16 weights at the start of every run. To skip that step, pre-export the grid once with `ltx2_export_int8_weights.py`, passing the same `--int8_weights_*` values, and add `--int8_weights_prequant` to the training command. The base checkpoint still loads normally; only the quantization is skipped.

```bash
python -m musubi_tuner.ltx2_export_int8_weights \
  --input_model /path/to/ltx-2.3-22b-dev.safetensors \
  --output_model /path/to/ltx-2.3-22b-dev.int8w.safetensors \
  --targets video --group_size 64 --convrot auto
```

Then add `--int8_weights_prequant /path/to/ltx-2.3-22b-dev.int8w.safetensors` to the training command. The sidecar's grid (`--group_size`, `--outlier_quantile`, `--sparse_ratio`, `--convrot`, `--dtype`) must match the `--int8_weights_*` flags or the run aborts. `--int8_weights_w8a8` is a runtime compute choice and is not baked into the sidecar, so the same matched sidecar can be used with or without W8A8 compute.

Measured int8 ConvRot rows (`--int8_weights --int8_weights_convrot auto --int8_weights_targets video --int8_weights_group_size 64`, Adafactor with fused backward, `--max_grad_norm 0`, no EMA in the measured path — the EMA shadow adds host RAM, not VRAM). Same measurement conventions and columns as the matrices above. Each row is `peak VRAM / s per optimizer step`:

| Resolution | Frames | video | av | v2v | v2v_av |
| --- | ---: | ---: | ---: | ---: | ---: |
| 832x480 | 49 | 18.3 GB / 3.86 s | 30.9 GB / 5.38 s | 18.7 GB / 3.80 s | 31.4 GB / 5.42 s |
| 832x480 | 65 | 19.0 GB / 4.45 s | 31.8 GB / 5.94 s | 19.3 GB / 4.38 s | 32.2 GB / 6.06 s |
| 832x480 | 81 | 19.7 GB / 4.85 s | 32.6 GB / 6.56 s | 20.0 GB / 4.99 s | 33.1 GB / 6.75 s |
| 832x480 | 97 | 20.3 GB / 5.47 s | 33.5 GB / 7.23 s | 20.6 GB / 5.65 s | 34.0 GB / 7.40 s |
| 832x480 | 113 | 21.0 GB / 6.08 s | 34.4 GB / 7.92 s | 21.3 GB / 6.23 s | 34.9 GB / 8.14 s |
| 832x480 | 129 | 21.6 GB / 6.74 s | 35.3 GB / 8.66 s | 22.0 GB / 6.89 s | 35.8 GB / 8.81 s |
| 832x480 | 145 | 22.3 GB / 7.36 s | 36.2 GB / 9.34 s | 22.6 GB / 7.50 s | 36.7 GB / 9.55 s |
| 832x480 | 161 | 23.0 GB / 8.04 s | 37.0 GB / 10.04 s | 23.3 GB / 8.17 s | 37.6 GB / 10.33 s |
| 832x480 | 193 | 24.3 GB / 9.43 s | 38.8 GB / 11.78 s | 24.6 GB / 9.60 s | 39.3 GB / 11.88 s |
| 832x480 | 241 | 26.3 GB / 11.66 s | 41.4 GB / 14.08 s | 26.6 GB / 11.86 s | 42.0 GB / 14.38 s |
| 1280x720 | 17 | 18.3 GB / 3.66 s | 30.9 GB / 5.30 s | 19.1 GB / 3.98 s | 31.9 GB / 5.66 s |
| 1280x720 | 33 | 19.9 GB / 5.05 s | 32.9 GB / 6.74 s | 20.7 GB / 5.40 s | 34.0 GB / 7.16 s |
| 1280x720 | 49 | 21.4 GB / 6.48 s | 35.0 GB / 8.30 s | 22.2 GB / 6.87 s | 36.0 GB / 8.83 s |
| 1280x720 | 65 | 23.0 GB / 8.02 s | 37.0 GB / 10.05 s | 23.8 GB / 8.41 s | 38.1 GB / 10.56 s |
| 1280x720 | 81 | 24.6 GB / 9.54 s | 39.1 GB / 11.77 s | 25.3 GB / 10.03 s | 40.1 GB / 12.33 s |
| 1280x720 | 97 | 26.1 GB / 11.20 s | 41.1 GB / 13.57 s | 26.9 GB / 11.60 s | 42.2 GB / 14.10 s |
| 1280x720 | 113 | 27.7 GB / 12.91 s | 43.1 GB / 15.38 s | 28.4 GB / 13.44 s | 44.2 GB / 16.05 s |
| 1280x720 | 129 | 29.2 GB / 14.67 s | 45.2 GB / 17.37 s | 30.0 GB / 15.20 s | 46.3 GB / 18.01 s |
| 1280x720 | 145 | 30.8 GB / 16.50 s | 47.2 GB / 19.36 s | 31.5 GB / 17.08 s | 48.3 GB / 20.14 s |
| 1280x720 | 161 | 32.3 GB / 18.44 s | 49.3 GB / 21.41 s | 33.1 GB / 19.12 s | 50.4 GB / 22.17 s |
| 1280x720 | 193 | 35.2 GB / 22.32 s | 53.0 GB / 25.77 s | 35.9 GB / 23.10 s | 54.1 GB / 26.42 s |
| 1280x720 | 241 | 39.8 GB / 28.82 s | 59.1 GB / 32.78 s | 40.5 GB / 29.90 s | 60.2 GB / 34.03 s |

### Optimizing VRAM Usage
<sub>[↑ contents](#table-of-contents)</sub>

Dense bf16 Adafactor video-only rows are around 54-66 GB, and AV rows add roughly 20 GB in the measured matrix. Standard block swap can bring short video-only dense bf16 training into the 24 GB class without quantizing the trainable weights. Quantized full-fine-tuning paths provide another memory/speed tradeoff. BAdam reduces active optimizer and gradient state, but the dense base transformer weights still stay resident and the full transformer forward still runs every step.

#### Block Swap for Full Fine-Tuning

Block swap keeps part of the transformer in system RAM and transfers each offloaded block to the GPU when it is needed. Full fine-tuning requires the standard bidirectional path because updated block weights must return to system RAM; `--block_swap_h2d_only` is only valid for frozen base weights such as LoRA training.

Measured 24 GB starting configuration for 512x512x33 video training:

```bash
--gradient_checkpointing \
--fused_backward_pass \
--max_grad_norm 0 \
--blocks_to_swap 16 \
--use_pinned_memory_for_block_swap
```

Measured workload: 512x512x33 video-only, batch size 1.

| Configuration | Peak VRAM | Peak host RAM | Median step time | Slowdown vs no swap |
| --- | ---: | ---: | ---: | ---: |
| No block swap | 45,811 MiB | 8.436 GiB | 1.850 s | 1.00x |
| 16 blocks, pinned | 23,045 MiB | 44.220 GiB | 2.705 s | 1.46x |
| 16 blocks, pinned + async backward | 30,616 MiB | 41.761 GiB | 2.493 s | 1.35x |
| 16 blocks, pinned + trainable ring | 22,738 MiB | 40.485 GiB | 1.972 s | 1.07x |

`--ltx2_block_swap_async_backward` improves pinned block swap by 7.9-8.0%. For fused-backward FFT, use `--ltx2_block_swap_trainable_ring` instead: it is 20.9% faster than the async path, stays within the 24 GB class, and is only 6.6% slower than no swap in this workload.

The measurements used an H100 with HBM3 VRAM, DDR5-4400 system RAM, and PCIe 5.0 x16. Pinned transfer bandwidth was 51.60 GiB/s H2D and 51.51 GiB/s D2H. GDDR6 is GPU memory and is not required for the host side; block swap uses ordinary system DDR memory. For similar transfer performance, use pinned DDR5 memory with a PCIe 5.0 x16 GPU link. PCIe 4.0, fewer PCIe lanes, slow system RAM, or cross-NUMA transfers will increase step time.

Start with 16 swapped blocks for 512x512x33 on a dedicated 24 GB GPU. Increase `--blocks_to_swap` when longer sequences, higher resolutions, AV training, display use, or other GPU processes require more headroom. The measured profile fits in 64 GB system RAM on a dedicated training system; 96-128 GB leaves additional headroom for the OS, data loading, and caching.

For quantized 24 GB-class alternatives, use a `Linear` replacement path:

- Start with Q-GaLore: `--qgalore_full_ft --qgalore_load_device cpu --qgalore_targets video --qgalore_rank 256`.
- Int8 weight storage is the other 24 GB-class route (see [Int8 / Int8 ConvRot / EMA](#int8--int8-convrot--ema)): with `--max_grad_norm 0`, short-context video-only training fits in roughly 21-23 GB while keeping a standard non-quantized optimizer through `--fused_backward_pass`; pair it with the EMA to recover sample fidelity, and budget the EMA's host-RAM cost (about two bytes per trainable parameter with `--ema_cpu_offload`).
- To test QAPOLLO in the same 24 GB-class envelope, start from the QAPOLLO command above and use `--qgalore_load_device cpu --qgalore_targets video` with `optim_bits=8`. QAPOLLO uses `QAPOLLOAdamW` and the `--apollo_*` optimizer settings instead of Q-GaLore's SVD projection settings.
- For `832x480x49`, expect roughly 21-23 GB on a 24 GB GPU, depending on rank, desktop VRAM use, and the local CUDA stack. Use `rank=128` for more headroom; try `rank=384` only after the lower-rank run leaves headroom.
- For longer contexts such as `512x512x65`, expect the run to sit close to the 24 GB limit and to be substantially slower than the short-context fit test. Start with T=33 or T=49 before scaling to T=65.
- Defer AV, 720p, training-time sampling, and frequent validation until a video-only baseline saves and resumes successfully on the target GPU.
- Dense `APOLLOAdamW` is not a 24 GB path because the base transformer weights remain dense. The 24 GB-class APOLLO route is QAPOLLO, with similar load/save constraints to Q-GaLore.
- Regular full-parameter checkpoint writes use the memory-efficient safetensors writer. Add `--qgalore_streaming_dequantize_save` if dense Q-GaLore/QAPOLLO checkpoint export exceeds available VRAM during saving; this uses more CPU work during saving. Merged checkpoints are large because they contain the fine-tuned base model, not an adapter.

Example Q-GaLore video-only command:

```bash
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train.py \
  --mixed_precision bf16 \
  --full_bf16 \
  --dataset_config dataset.toml \
  --ltx2_checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --flash_attn \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --max_data_loader_n_workers 1 \
  --persistent_data_loader_workers \
  --learning_rate 1e-5 \
  --optimizer_type QGaLoreAdamW8bit \
  --optimizer_args weight_decay=0.0 min_8bit_size=4096 \
  --qgalore_full_ft \
  --qgalore_load_device cpu \
  --qgalore_targets video \
  --qgalore_rank 256 \
  --qgalore_update_proj_gap 200 \
  --qgalore_scale 0.25 \
  --qgalore_proj_quant \
  --qgalore_proj_bits 4 \
  --qgalore_proj_group_size 256 \
  --qgalore_svd_method lowrank \
  --qgalore_svd_oversampling 32 \
  --qgalore_svd_niter 1 \
  --qgalore_weight_bits 8 \
  --qgalore_weight_group_size 0 \
  --qgalore_stochastic_round \
  --max_grad_norm 0 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --timestep_sampling shifted_logit_normal \
  --fused_backward_pass \
  --qgalore_streaming_dequantize_save \
  --save_every_n_steps 5000 \
  --save_state \
  --save_state_on_train_end \
  --max_train_steps 50000 \
  --output_dir output/full_ft_qgalore_24gb \
  --output_name ltx23_qgalore_24gb_video
```

For longer runs, start with video-only short-context training until checkpoint save and resume have been validated on the target GPU.

**FP8 full fine-tuning (`--fp8_gemm`, experimental).** Replaces eligible attention/FFN `Linear` layers with FP8 forward/backward calls to the private `torch._scaled_mm` API while retaining bf16 master weights. The runtime gate requires CUDA compute capability ≥ 8.9. `--fp8_gemm_compile` defaults on and requests regional compilation; if compilation raises, the wrapper logs a warning and continues with the eager FP8 wrapper. `--fp8_gemm_scaling tensor` (default) uses one dynamic scale per input tensor; `rowwise` passes per-row/per-column scale tensors to the installed build's private `torch._scaled_mm` implementation. Support for that private rowwise form is build-dependent. Other flags are `--fp8_gemm_targets`, `--fp8_gemm_grad_dtype {e4m3,e5m2}`, and `--fp8_gemm_min_numel`. The path is mutually exclusive with `--qgalore_full_ft`, `--fp8_scaled`, and `--ltx2_model_parallel`. No fixed memory, speed, numerical-stability, or output-quality claim is made; measure and validate the exact target build and workload.

> [!CAUTION]
> **Experimental.** Validate loss, long-run stability, and final sample quality on your own runs (and monitor gradient norms) before relying on it for production.

**Gradient noise scale probe (batch-size estimate).** `--noise_scale_probe K` runs a diagnostic instead of training: it estimates the gradient noise scale ([McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)) and prints the critical batch `B_simple = tr(Σ)/|G|²` before exiting — the batch size below which raising the batch cuts gradient noise roughly linearly and above which returns diminish. Run it **without** `--fused_backward_pass` (the probe needs `.grad` to persist after `backward()`), and because the full bf16 gradient and the weights do not co-reside, use `--noise_scale_probe_param_frac` (e.g. `0.25`–`0.5`) to measure over a stratified parameter subset; `--noise_scale_probe_rounds` median-averages across rounds. Read it as an order of magnitude (it rises as training progresses), then set the effective batch (`micro_bs × gradient_accumulation_steps × num_gpus`) near `B_simple`. Append to the Adafactor command, dropping `--fused_backward_pass`: `--noise_scale_probe 96 --noise_scale_probe_rounds 8 --noise_scale_probe_param_frac 0.34`.

---

## References
<sub>[↑ contents](#table-of-contents)</sub>

**Musubi Tuner Documentation**
- [Dataset Configuration](./dataset_config.md) — TOML format, `frame_extraction` modes, JSONL metadata, control images, resolution bucketing
- [Sampling During Training](./sampling_during_training.md) — Prompt file format, per-prompt guidance scale, negative prompts, sampling CLI flags
- [Advanced Configuration](./advanced_config.md) — `--config_file` TOML training configuration, `--network_args` format, LoRA+, TensorBoard/WandB logging, PyTorch Dynamo, timestep bucketing, Schedule-Free optimizer
- [Tools](./tools.md) — Post-hoc EMA LoRA merging, image captioning with Qwen2.5-VL
- [LoHa/LoKr](./loha_lokr.md) — Alternative parameter-efficient fine-tuning methods
- [torch.compile](./torch_compile.md) — PyTorch JIT compilation for faster training and inference
- [LyCORIS Algorithm List](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md) and [Guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md) — LoKR, LoHA, LoCoN and other algorithm details (used via `pip install lycoris-lora`)

**Research**
- [Explorative Modeling (arXiv 2607.27372)](https://arxiv.org/abs/2607.27372) — Forward XM hard-min exploration over multiple noise candidates; basis for `--ltx2_xm_k`
- ID-LoRA — In-context identity LoRA; the audio-reference IC-LoRA implementation in this trainer is based on this approach
- [Adaptive LoRA Ranks / "Not All Layers Are Created Equal" (arXiv 2603.21884)](https://arxiv.org/abs/2603.21884) — Adaptive per-layer LoRA rank allocation; basis for the adaptive LoRA rank feature
- [Adafactor (ICML 2018)](https://proceedings.mlr.press/v80/shazeer18a.html) — Factored second-moment optimizer-state background for the Adafactor full-parameter path
- [BAdam (arXiv 2404.02827)](https://arxiv.org/abs/2404.02827) — Block-coordinate Adam-style full-parameter optimization background for the BAdam path
- [BREAD / Landscape Correction (OpenReview)](https://openreview.net/forum?id=zs6bRl05g8) — Block-coordinate landscape-correction background for the BREAD-SGD ablation
- [GaLore (arXiv 2403.03507)](https://arxiv.org/abs/2403.03507) — Gradient low-rank projection background for Q-GaLore-style optimizer-state reduction
- [Q-GaLore (arXiv 2407.08296)](https://arxiv.org/abs/2407.08296) — Quantized low-rank gradient-projection background for `--qgalore_full_ft`
- [APOLLO (arXiv 2412.05270)](https://arxiv.org/abs/2412.05270) — Low-rank optimizer-state background for `APOLLOAdamW` and `QAPOLLOAdamW`
- [QLoRA (arXiv 2305.14314)](https://arxiv.org/abs/2305.14314) — Introduces NF4 quantization used by the `--nf4_base` implementation
- [LoftQ (arXiv 2310.08659)](https://arxiv.org/abs/2310.08659) — Quantization-aware LoRA initialization used by `--loftq_init`
- [AWQ (arXiv 2306.00978)](https://arxiv.org/abs/2306.00978) — Activation-aware quantization background for `--awq_calibration`
- [FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models (arXiv 2607.05711)](https://arxiv.org/abs/2607.05711) — Research background for the local `--w4a4g4` design; the local path is not claimed to reproduce the paper
- [DINOv2 (arXiv 2304.07193)](https://arxiv.org/abs/2304.07193) — External visual features used by CREPA dino mode
- [CREPA (arXiv 2506.09229)](https://arxiv.org/abs/2506.09229) — Cross-frame Representation Alignment; basis for CREPA dino mode (`--crepa` with `--crepa_args mode=dino`, DINOv2 teacher from neighboring frames)
- [Latent Temporal Discrepancy (arXiv 2601.20504)](https://arxiv.org/abs/2601.20504) — Motion-prior loss weighting basis for `--latent_temporal_weighting`
- [CCL / TARP / DCR (arXiv 2603.18600)](https://arxiv.org/abs/2603.18600) — Cross-modal context learning; basis for `--tarp` and `--dcr`
- [Self-Flow (arXiv 2603.06507)](https://arxiv.org/abs/2603.06507) — Self-supervised flow matching regularization; basis for `--self_flow`
- [ViBe / HFATO / Relay LoRA (arXiv 2603.23326)](https://arxiv.org/abs/2603.23326) — Basis for `--hfato` and the Relay LoRA workflow
- [G2D (arXiv 2506.21514)](https://arxiv.org/abs/2506.21514) — Sequential Modality Prioritization inspiration for modality freezing
- [Omni-Customizer (arXiv 2605.17488)](https://arxiv.org/abs/2605.17488) — Interleaved modality-decoupled training inspiration for the AV curriculum controls; the configurable loss-policy implementation in this trainer is an independent generalization
- [UniAVGen (arXiv 2511.03334)](https://arxiv.org/abs/2511.03334) — Joint audio-video generation reference for lower-LR AV training guidance
- [Harmony (arXiv 2511.21579)](https://arxiv.org/abs/2511.21579) — Cross-Task Synergy; basis for `--cts_lambda_video_driven` and `--cts_lambda_audio_driven`
- [OmniNFT](https://github.com/zghhui/OmniNFT) — DiffusionNFT-based audio-video RL project; initial reference for the NFT update lineage and several reward checkpoints
- [DeepSeekMath / GRPO (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300) — Shao et al.; group-relative policy optimization, the basis for the per-prompt group-relative advantages
- [Flow-GRPO (arXiv 2505.05470)](https://arxiv.org/abs/2505.05470) — Online policy-gradient RL for flow-matching models via ODE→SDE conversion; background for the `--rl_sde_sampler` trajectory sampling
- [DanceGRPO (arXiv 2505.07818)](https://arxiv.org/abs/2505.07818) — GRPO applied across visual generation paradigms including video models; field background for video RL
- [DDPO (arXiv 2305.13301)](https://arxiv.org/abs/2305.13301) — Black et al., "Training Diffusion Models with Reinforcement Learning"; policy-gradient RL for diffusion samplers background for the `ppo` rule
- [DPOK (arXiv 2305.16381)](https://arxiv.org/abs/2305.16381) — Fan et al.; KL-regularized RL fine-tuning of diffusion models background for the KL-anchored update
- [Diffusion-DPO (arXiv 2311.12908)](https://arxiv.org/abs/2311.12908) — Wallace et al.; basis for the `dpo` (`--rl_loss dpo`) preference update
- [PPO (arXiv 1707.06347)](https://arxiv.org/abs/1707.06347) — Schulman et al., "Proximal Policy Optimization Algorithms"; basis for the clipped-ratio objective in the `ppo` rule
- RWR — Peters & Schaal (2007), "Reinforcement learning by reward-weighted regression" (ICML); basis for the `rwr` (`--rl_loss rwr`) update

**LTX Resources**
- [LTX-2](https://github.com/Lightricks/LTX-2) — LTX-2/2.3 model and pipeline resources
- [LTX Documentation](https://docs.ltx.video/open-source-model/getting-started/overview) — Unified docs hub: open-source model, API reference, ComfyUI integration, LoRA usage, and LTX-2 trainer guide
- [LTX Discord](https://discord.gg/epQmav9hAR) — Official LTX team Discord

**Alternative Trainers**
- [ai-toolkit](https://github.com/ostris/ai-toolkit) (ostris) — General diffusion fine-tuning toolkit with LTX-2/2.3 LoRA support; slider LoRA training is based on its implementation
- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) — Pipeline-parallel diffusion model trainer with LTX-Video and LTX 2.3 support; initial LTX 2.3 support covers T2I/T2V training, without audio
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — ModelScope diffusion synthesis framework with LTX-2/2.3 support
- [SimpleTuner](https://github.com/bghira/SimpleTuner) — Multi-model fine-tuning framework with LTX-2/2.3 support; CREPA backbone mode (`--crepa_args mode=backbone`) is inspired by its LayerSync regularizer
- [rs-nodes](https://github.com/richservo/rs-nodes) — ComfyUI node pack with an in-process LTX-2 LoRA trainer that reuses the already-loaded transformer; supports subject/style/motion training with divergence detection and quantization options

**Community Resources**
- [awesome-ltx2](https://github.com/wildminder/awesome-ltx2) — Curated list of LTX-2 resources, tools, models, and guides
- [Banodoco Discord](https://discord.gg/BdvB5E7rsZ) — Active AI video generation community; discussions on LTX-2 training, workflows, and research
- [Windows Installation Guide](https://github.com/AkaneTendo25/musubi-tuner/discussions/19) — Windows-specific setup (Python 3.12, CUDA, Flash Attention 2), dependencies, troubleshooting
- [LTX-2 Training Optimizers](https://github.com/AkaneTendo25/musubi-tuner/discussions/21) — Optimizer comparison for LTX-2 training: AdamW, Prodigy, Muon, CAME, and recommended settings
- [LTX-2 Audio Dataset Builder](https://github.com/dorpxam/LTX-2-Audio-Dataset-Builder) — Tool to automate audio dataset creation: transforms raw audio into clean, captioned segments optimized for LTX-2 audio-only training

**Tutorials & Guides**
- [LTX-2 LoRA Training Complete Guide](https://apatero.com/blog/ltx-2-lora-training-fine-tuning-complete-guide-2025) (Apatero) — Dataset preparation, training configuration, and LoRA deployment walkthrough
- [How to Train a LTX-2 Character LoRA](https://ghost.oxen.ai/how-to-train-a-ltx-2-character-lora-with-oxen-ai/) (Oxen.ai) — Character-consistency LoRA training with dataset prep tips for audio clips

**Cloud Platforms**
- [fal.ai LTX-2 Trainer](https://fal.ai/models/fal-ai/ltx2-video-trainer) — Cloud-based LTX-2 LoRA training via API (~$0.005/step)
- [WaveSpeedAI LTX-2](https://wavespeed.ai/landing/ltx2) — Hosted LTX-2 inference (T2V, I2V, video extend, lipsync)
