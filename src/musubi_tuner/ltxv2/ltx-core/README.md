# LTX-Core

The foundational library for the LTX-2 Audio-Video generation model. This package contains the raw model definitions, component implementations, and loading logic used by `ltx-pipelines` and `ltx-trainer`.

## 📦 What's Inside?

- **`components/`**: Modular diffusion components (Schedulers, Guiders, Noisers, Patchifiers) following standard protocols
- **`conditioning/`**: Tools for preparing latent states and applying conditioning (image, video, keyframes)
- **`guidance/`**: Perturbation system for fine-grained control over attention mechanisms
- **`loader/`**: Utilities for loading weights from `.safetensors`, fusing LoRAs, and managing memory
- **`model/`**: PyTorch implementations of the LTX-2 Transformer, Video VAE, Audio VAE, Vocoder and Upscaler
- **`text_encoders/gemma`**: Gemma text encoder implementation with tokenizers, feature extractors, and separate encoders for audio-video and video-only generation

## 🚀 Quick Start

`ltx-core` provides the building blocks (models, components, and utilities) needed to construct inference flows. For ready-made inference pipelines use [`ltx-pipelines`](../ltx-pipelines/) or [`ltx-trainer`](../ltx-trainer/) for training.

## 🔧 Installation

```bash
# From the repository root
uv sync --frozen

# Or install as a package
pip install -e packages/ltx-core
```

## Building Blocks Overview

`ltx-core` provides modular components that can be combined to build custom inference flows:

### Core Models

- **Transformer** ([`model/transformer/`](src/ltx_core/model/transformer/)): The 48-layer LTX-2 transformer with cross-modal attention for joint audio-video processing. Expects inputs in [`Modality`](src/ltx_core/model/transformer/modality.py) format
- **Video VAE** ([`model/video_vae/`](src/ltx_core/model/video_vae/)): Encodes/decodes video pixels to/from latent space with temporal and spatial compression
- **Audio VAE** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Encodes/decodes audio spectrograms to/from latent space
- **Vocoder** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/)): Neural vocoder that converts mel spectrograms to audio waveforms
- **Text Encoder** ([`text_encoders/`](src/ltx_core/text_encoders/)): Gemma-based encoder that produces separate embeddings for video and audio conditioning
- **Spatial Upscaler** ([`model/upsampler/`](src/ltx_core/model/upsampler/)): Upsamples latent representations for higher-resolution generation

### Diffusion Components

- **Schedulers** ([`components/schedulers.py`](src/ltx_core/components/schedulers.py)): Noise schedules (LTX2Scheduler, LinearQuadratic, Beta) that control the denoising process
- **Guiders** ([`components/guiders.py`](src/ltx_core/components/guiders.py)): Guidance strategies (CFG, STG, APG) for controlling generation quality and adherence to prompts
- **Noisers** ([`components/noisers.py`](src/ltx_core/components/noisers.py)): Add noise to latents according to the diffusion schedule
- **Patchifiers** ([`components/patchifiers.py`](src/ltx_core/components/patchifiers.py)): Convert between spatial latents `[B, C, F, H, W]` and sequence format `[B, seq_len, dim]` for transformer processing

### Conditioning & Control

- **Conditioning** ([`conditioning/`](src/ltx_core/conditioning/)): Tools for preparing and applying various conditioning types (image, video, keyframes)
- **Guidance** ([`guidance/`](src/ltx_core/guidance/)): Perturbation system for fine-grained control over attention mechanisms (e.g., skipping specific attention layers)

### Utilities

- **Loader** ([`loader/`](src/ltx_core/loader/)): Model loading from `.safetensors`, LoRA fusion, weight remapping, and memory management

For complete, production-ready pipeline implementations that combine these building blocks, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

# Architecture Overview

This section provides a deep dive into the internal architecture of the LTX-2 Audio-Video generation model.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The Transformer](#the-transformer)
3. [Video VAE](#video-vae)
4. [Audio VAE](#audio-vae)
5. [Text Encoding (Gemma)](#text-encoding-gemma)
6. [Spatial Upscaler](#spatial-upsampler)
7. [Data Flow](#data-flow)

---

## High-Level Architecture

LTX-2 is a **joint Audio-Video diffusion transformer** that processes both modalities simultaneously in a unified architecture. Unlike traditional models that handle video and audio separately, LTX-2 uses cross-modal attention to enable natural synchronization.

```text
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                        │
│                                                             │
│  Video Pixels → Video VAE Encoder → Video Latents           │
│  Audio Waveform → Audio VAE Encoder → Audio Latents         │
│  Text Prompt → Gemma Encoder → Text Embeddings              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LTX-2 TRANSFORMER (48 Blocks)                  │
│                                                             │
│  ┌──────────────┐              ┌──────────────┐             │
│  │ Video Stream │              │ Audio Stream │             │
│  │              │              │              │             │
│  │ Self-Attn    │              │ Self-Attn    │             │
│  │ Cross-Attn   │              │ Cross-Attn   │             │
│  │              │◄────────────►│              │             │
│  │ A↔V Cross    │              │ A↔V Cross    │             │
│  │ Feed-Forward │              │ Feed-Forward │             │
│  └──────────────┘              └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT DECODING                          │
│                                                             │
│  Video Latents → Video VAE Decoder → Video Pixels           │
│  Audio Latents → Audio VAE Decoder → Mel Spectrogram        │
│  Mel Spectrogram → Vocoder → Audio Waveform                 │
└─────────────────────────────────────────────────────────────┘
```

---

## The Transformer

The core of LTX-2 is a 48-layer transformer that processes both video and audio tokens simultaneously.

### Model Structure

**Source**: [`src/ltx_core/model/transformer/model.py`](src/ltx_core/model/transformer/model.py)

The `LTXModel` class implements the transformer. It supports both video-only and audio-video generation modes. For actual usage, see the [`ltx-pipelines`](../ltx-pipelines/) package which handles model loading and initialization.

### Transformer Block Architecture

**Source**: [`src/ltx_core/model/transformer/transformer.py`](src/ltx_core/model/transformer/transformer.py)

```text
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                        │
│                                                             │
│  VIDEO PATH:                                                │
│    Input → RMSNorm → AdaLN → Self-Attn (attn1)              │
│         → RMSNorm → Cross-Attn (attn2, text)                │
│         → RMSNorm → AdaLN → A↔V Cross-Attn                  │
│         → RMSNorm → AdaLN → Feed-Forward (ff) → Output      │
│                                                             │
│  AUDIO PATH:                                                │
│    Input → RMSNorm → AdaLN → Self-Attn (audio_attn1)        │
│         → RMSNorm → Cross-Attn (audio_attn2, text)          │
│         → RMSNorm → AdaLN → A↔V Cross-Attn                  │
│         → RMSNorm → AdaLN → Feed-Forward (audio_ff)         │
│                                                             │
│  AdaLN (Adaptive Layer Normalization):                      │
│    - Uses scale_shift_table (6 params) for video/audio      │
│    - Uses scale_shift_table_a2v_ca (5 params) for A↔V CA    │
│    - Conditioned on per-token timestep embeddings           │
└─────────────────────────────────────────────────────────────┘
```

### Perturbations

The transformer supports [**perturbations**](src/ltx_core/guidance/perturbations.py) that selectively skip attention operations.

Perturbations allow you to disable specific attention mechanisms during inference, which is useful for guidance techniques like STG (Spatio-Temporal Guidance).

**Supported Perturbation Types**:

- `SKIP_VIDEO_SELF_ATTN`: Skip video self-attention
- `SKIP_AUDIO_SELF_ATTN`: Skip audio self-attention
- `SKIP_A2V_CROSS_ATTN`: Skip audio-to-video cross-attention
- `SKIP_V2A_CROSS_ATTN`: Skip video-to-audio cross-attention

Perturbations are used internally by guidance mechanisms like STG (Spatio-Temporal Guidance). For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Video VAE

The Video VAE ([`src/ltx_core/model/video_vae/`](src/ltx_core/model/video_vae/)) encodes video pixels into latent representations and decodes them back.

### Architecture

- **Encoder**: Compresses `[B, 3, F, H, W]` pixels → `[B, 128, F', H/32, W/32]` latents
  - Where `F' = 1 + (F-1)/8` (frame count must satisfy `(F-1) % 8 == 0`)
  - Example: `[B, 3, 33, 512, 512]` → `[B, 128, 5, 16, 16]`
- **Decoder**: Expands `[B, 128, F, H, W]` latents → `[B, 3, F', H*32, W*32]` pixels
  - Where `F' = 1 + (F-1)*8`
  - Example: `[B, 128, 5, 16, 16]` → `[B, 3, 33, 512, 512]`

The Video VAE is used internally by pipelines for encoding video pixels to latents and decoding latents back to pixels. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Audio VAE

The Audio VAE ([`src/ltx_core/model/audio_vae/`](src/ltx_core/model/audio_vae/)) processes audio spectrograms.

### Audio VAE Architecture

- **Encoder**: Compresses mel spectrogram `[B, mel_bins, T]` → `[B, 8, T/4, 16]` latents
  - Temporal downsampling: 4× (`LATENT_DOWNSAMPLE_FACTOR = 4`)
  - Frequency bins: Fixed 16 mel bins in latent space
  - Latent channels: 8
- **Decoder**: Expands `[B, 8, T, 16]` latents → mel spectrogram `[B, mel_bins, T*4]`
- **Vocoder**: Converts mel spectrogram → audio waveform

**Downsampling**:

- Temporal: 4× (time steps)
- Frequency: Variable (input mel_bins → fixed 16 in latent space)

The Audio VAE is used internally by pipelines for encoding mel spectrograms to latents and decoding latents back to mel spectrograms. The vocoder converts mel spectrograms to audio waveforms. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Text Encoding (Gemma)

LTX-2 uses **Gemma** (Google's open LLM) as the text encoder, located in [`src/ltx_core/text_encoders/gemma/`](src/ltx_core/text_encoders/gemma/).

### Text Encoder Architecture

- **Tokenizer**: Converts text → token IDs
- **Gemma Model**: Processes tokens → embeddings
- **Text Projection**: Uses `PixArtAlphaTextProjection` to project caption embeddings
  - Two-layer MLP with GELU (tanh approximation) or SiLU activation
  - Projects from caption channels (3840) to model dimensions
- **Feature Extractor**: Extracts video/audio-specific embeddings
- **Separate Encoders**:
  - `AVEncoder`: For audio-video generation (outputs separate video and audio contexts)
  - `VideoOnlyEncoder`: For video-only generation

### System Prompts

System prompts are also used to enhance user's prompts.

- **Text-to-Video**: [`gemma_t2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt)
- **Image-to-Video**: [`gemma_i2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt)

**Important**: Video and audio receive **different** context embeddings, even from the same prompt. This allows better modality-specific conditioning.

**Output Format**:

- Video context: `[B, seq_len, 4096]` - Video-specific text embeddings
- Audio context: `[B, seq_len, 2048]` - Audio-specific text embeddings

The text encoder is used internally by pipelines. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Upscaler

The Upscaler ([`src/ltx_core/model/upsampler/`](src/ltx_core/model/upsampler/)) upsamples latent representations for higher-resolution output.

The spatial upsampler is used internally by two-stage pipelines (e.g., [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py), [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py)) to upsample low-resolution latents before final VAE decoding. For usage examples, see the [`ltx-pipelines`](../ltx-pipelines/) package.

---

## Data Flow

### Complete Generation Pipeline

Here's how all the components work together conceptually ([`src/ltx_core/components/`](src/ltx_core/components/)):

**Pipeline Steps**:

1. **Text Encoding**: Text prompt → Gemma encoder → separate video/audio embeddings
2. **Latent Initialization**: Initialize noise latents in spatial format `[B, C, F, H, W]`
3. **Patchification**: Convert spatial latents to sequence format `[B, seq_len, dim]` for transformer
4. **Sigma Schedule**: Generate noise schedule (adapts to token count)
5. **Denoising Loop**: Iteratively denoise using transformer predictions
   - Create Modality inputs with per-token timesteps and RoPE positions
   - Forward pass through transformer (conditional and unconditional for CFG)
   - Apply guidance (CFG, STG, etc.)
   - Update latents using diffusion step (Euler, etc.)
6. **Unpatchification**: Convert sequence back to spatial format
7. **VAE Decoding**: Decode latents to pixel space (with optional upsampling for two-stage)

- [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py) - Two-stage text-to-video (recommended)
- [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py) - Video-to-video with IC-LoRA control
- [`DistilledPipeline`](../ltx-pipelines/src/ltx_pipelines/distilled.py) - Fast inference with distilled model
- [`KeyframeInterpolationPipeline`](../ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py) - Keyframe-based interpolation

See the [ltx-pipelines README](../ltx-pipelines/README.md) for usage examples.

## 🔗 Related Projects

- **[ltx-pipelines](../ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and video-to-video
- **[ltx-trainer](../ltx-trainer/)** - Training and fine-tuning tools
