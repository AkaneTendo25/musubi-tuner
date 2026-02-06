# LTX-2

Status: LTX-2 support is work in progress and may be incomplete or unstable.

This guide details the process for training LTX-2 LoRA models:
1. **Caching Latents** (Video/Image + Audio)
2. **Caching Text Encoder Outputs** (Prompts)
3. **Training**
4. **Slider LoRA Training** (Controllable direction sliders)

## Installation

See the [Installation Guide](https://github.com/AkaneTendo25/musubi-tuner/discussions/19) for detailed setup instructions (Windows/Linux, dependencies, flash-attn, troubleshooting).

---

## Supported Dataset Types

| Mode | Dataset Type | Notes |
|------|--------------|-------|
| `video` | Images | Treated as 1-frame samples (`F=1`) |
| `video` | Videos | Standard video training |
| `av` | Videos with audio | Audio extracted from video or external audio files |
| `audio` | Audio only | Dataset must be audio-only; video latents are dummy placeholders |

---

## 1. Caching Latents

This step pre-processes media files into VAE latents to speed up training.

**Script:** `ltx2_cache_latents.py`

### Example Command
```bash
python ltx2_cache_latents.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --device cuda ^
  --vae_dtype bf16 ^
  --ltx2_mode av ^
  --ltx2_audio_source video
```

### Key Arguments
- `--ltx2_mode av`: Enables Audio-Video processing. Caches both `*_ltx2.safetensors` (video) and `*_ltx2_audio.safetensors` (audio) latents. `--ltx_mode` is accepted as an alias.
- `--ltx2_audio_source video|audio_files`: Use audio from the video or from external files.
- `--ltx2_audio_dir`, `--ltx2_audio_ext`: Optional when using `--ltx2_audio_source audio_files` (default extension: `.wav`).
- `--ltx2_checkpoint`: Required for `--ltx2_mode av` or `--ltx2_mode audio`.
- `--vae_dtype`: Data type for VAE latents (default comes from the cache script).

### Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2.safetensors` | Video latents: `latents_{F}x{H}x{W}_{dtype}` |
| `*_ltx2_audio.safetensors` | Audio latents: `audio_latents_{T}x{mel_bins}x{channels}_{dtype}`, `audio_lengths_int32` |

### Memory Optimization for Caching
If you encounter Out-Of-Memory (OOM) errors during caching (especially with higher resolutions like 1080p), use VAE temporal chunking:

```bash
python ltx2_cache_latents.py ^
  ...
  --vae_chunk_size 16
```

- `--vae_chunk_size`: Processes video in temporal chunks (e.g., 16 or 32 frames at a time). Default: `None` (all frames).

---

## 2. Caching Text Encoder Outputs

This step pre-computes text embeddings using the Gemma text encoder.

**Script:** `ltx2_cache_text_encoder_outputs.py`

### Example Command
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

### Key Arguments
- `--gemma_root`: Path to the local Gemma model folder (Hugging Face format).
- `--gemma_load_in_8bit`: Loads Gemma in 8-bit quantization.
- `--gemma_load_in_4bit`: Loads Gemma in 4-bit quantization.
- `--ltx2_checkpoint`: Required. Use `--ltx2_text_encoder_checkpoint` to override for text encoder connector weights.
- `--ltx2_mode av`: MUST match the mode used in latent caching. Concatenates video and audio prompt embeddings. `--ltx_mode` is accepted as an alias.
- 8-bit/4-bit loading requires `--device cuda`.

### Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2_te.safetensors` | `video_prompt_embeds_{dtype}`, `audio_prompt_embeds_{dtype}` (av only), `prompt_attention_mask`, `text_{dtype}`, `text_mask` |

---

## 3. Training

Launch the training loop using `accelerate`.

**Script:** `ltx2_train_network.py`

### Example Command
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --gemma_load_in_8bit ^
  --gemma_root /path/to/gemma ^
  --separate_audio_buckets ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --ltx2_mode av ^
  --fp8_base ^
  --fp8_scaled ^
  --blocks_to_swap 10 ^
  --flash_attn ^
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
  --output_name ltx2_lora
```

### Key Arguments

#### Memory Optimization
- `--fp8_base`, `--fp8_scaled`: Casts base model weights to FP8.
- `--blocks_to_swap X`: Offloads X transformer blocks to CPU (max 47 for 48-block model). Higher values save more VRAM but increase CPU↔GPU overhead.
- `--use_pinned_memory_for_block_swap`: Uses pinned memory for faster CPU↔GPU block transfers.
- `--gradient_checkpointing`: Reduces VRAM by recomputing activations during backward pass.
- `--gradient_checkpointing_cpu_offload`: Offloads activations to CPU during gradient checkpointing recomputation.
- `--ffn_chunk_target all|video|audio`: Enable FFN chunking for selected modules. Reduces VRAM by processing FFN in chunks.
- `--ffn_chunk_size N`: Chunk size for FFN chunking (0 disables).
- `--split_attn_target all|self|cross|...`: Enable split attention for selected attention modules.
- `--split_attn_mode batch|query`: Split attention by batch dimension or query length.
- `--split_attn_chunk_size N`: Chunk size for query-based split attention (0 uses default 1024).

#### Aggressive VRAM Optimization (8-16GB GPUs)

For maximum VRAM savings on 8-16GB GPUs, use this combination of flags:

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
  --flash_attn ^
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
- Use `--fp8_base --fp8_scaled`
- Use `--blocks_to_swap 47` (keeps only 1 block on GPU)
- Use smaller LoRA rank (`--network_dim 16` instead of 32)
- Use smaller training resolutions (e.g., 512x320)
- Reduce `--sample_vae_temporal_tile_size` to 24 or lower
- Use `--use_pinned_memory_for_block_swap` - faster transfers

#### Audio-Video Support
- `--ltx2_mode av`: Enables joint Video+Audio training logic.
- `--separate_audio_buckets`: Keeps audio and non-audio items in separate batches (reduces VRAM for image/video-only batches).

#### Loss Weighting
- `--video_loss_weight`: Weight for video loss (default: 1.0).
- `--audio_loss_weight`: Weight for audio loss in AV mode (default: 1.0).

#### Timestep Sampling
- `--timestep_sampling shifted_logit_normal`: Default LTX-2 method. Uses a shifted logit-normal distribution where the shift is computed based on sequence length (frames × height × width).
- `--timestep_sampling uniform`: Simple uniform sampling from [0, 1]. Alternative if you want simpler behavior.
- `--logit_std`: Standard deviation for the logit-normal distribution (default: 1.0). Only used with `shifted_logit_normal`.
- `--min_timestep` / `--max_timestep`: Optional timestep range constraints.

**Note:** The `shifted_logit_normal` shift is linearly interpolated from 0.95 (at 1024 tokens) to 2.05 (at 4096 tokens) based on sequence length.

#### LoRA Targets
Use `--lora_target_preset` to control which layers LoRA targets:

| Preset | Layers | Use Case |
|--------|--------|----------|
| `t2v` (default) | Attention only (`to_q`, `to_k`, `to_v`, `to_out.0`) | Text-to-video, matches official LTX-2 trainer |
| `v2v` | Attention + FFN | Video-to-video / IC-LoRA style |
| `audio` | Audio attention/FFN + audio-side cross-modal attention | Audio-only training (auto-selected when `--ltx2_mode audio`) |
| `full` | All linear layers | Maximum expressiveness, larger file size |

All presets apply to all relevant attention types: self-attention, cross-attention, and cross-modal attention (in AV mode). Connector layers are always excluded.

To use custom layer patterns instead of a preset, use `--network_args`:
```bash
--network_args "include_patterns=['.*\.to_k$','.*\.to_q$','.*\.to_v$','.*\.to_out\.0$','.*\.ff\.net\.0\.proj$','.*\.ff\.net\.2$']"
```
Custom `include_patterns` override any preset.

**Note:** IC-LoRA training is not currently supported but is being developed.

#### Sampling with Tiled VAE
- `--sample_tiled_vae`: Enable tiled VAE decoding during sampling to reduce VRAM usage.
- `--sample_vae_tile_size 512`: Spatial tile size.
- `--sample_vae_tile_overlap 64`: Spatial overlap (pixels).
- `--sample_vae_temporal_tile_size 0`: Temporal tile size (0 disables temporal tiling).
- `--sample_vae_temporal_tile_overlap 8`: Temporal overlap (frames).
- `--sample_merge_audio`: Merges generated audio into the `.mp4`.

#### Precached Sample Prompts
To avoid loading Gemma during training for sample generation, you can precache the prompt embeddings:

1. During text encoder caching, add `--precache_sample_prompts --sample_prompts sampling_prompts.txt` to also cache the sample prompt embeddings.
2. During training, add `--use_precached_sample_prompts` (or `--precache_sample_prompts`) to load embeddings from cache instead of running Gemma.
- `--sample_prompts_cache`: Path to the precached embeddings file. Defaults to `<cache_directory>/ltx2_sample_prompts_cache.pt`.

#### Two-Stage Sampling (WIP)
Two-stage inference generates at half resolution, then upsamples and refines for better quality. This feature is work in progress and may not produce optimal results yet. Disabled by default.

- `--sample_two_stage`: Enable two-stage inference during sampling.
- `--spatial_upsampler_path`: Path to spatial upsampler model (e.g., `ltx-2-spatial-upscaler-x2-1.0.safetensors`). Required when `--sample_two_stage` is set.
- `--distilled_lora_path`: Path to distilled LoRA (e.g., `ltx-2-19b-distilled-lora-384.safetensors`) for stage 2 refinement. Optional.
- `--sample_stage2_steps`: Number of denoising steps for stage 2 refinement (default: 3).

---

## Dataset Configuration

The dataset config is a TOML file with `[general]` defaults and `[[datasets]]` entries.

### Video Dataset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `video_directory` | string | — | Path to video/image directory |
| `video_jsonl_file` | string | — | Path to JSONL metadata file |
| `resolution` | int or [int, int] | [960, 544] | Target resolution |
| `target_frames` | [int] | [1] | List of target frame counts |
| `frame_extraction` | string | `"head"` | Frame extraction mode |
| `max_frames` | int | 129 | Maximum number of frames |
| `source_fps` | float | auto-detected | Source video FPS. Auto-detected from video container metadata when not set. Use this to override auto-detection. |
| `target_fps` | float | 24.0 | Target training FPS. Frames are resampled to this rate. When audio is present and the source video has a different FPS, the audio waveform is automatically time-stretched (pitch-preserving) to match the target video duration. |
| `batch_size` | int | 1 | Batch size |
| `num_repeats` | int | 1 | Dataset repetitions |
| `enable_bucket` | bool | false | Enable resolution bucketing |
| `cache_directory` | string | — | Latent cache output directory |
| `separate_audio_buckets` | bool | false | Keep audio/non-audio items in separate batches |

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
target_fps = 24    # optional, defaults to 24
```

### Frame Rate (FPS) Handling

LTX-2 was trained on 24fps video. During latent caching, the source FPS is **auto-detected** from each video's container metadata and frames are resampled to `target_fps` (default: 24). This ensures the model sees video at the correct temporal rate regardless of the source material.

#### How It Works

1. For each video file, the source FPS is read from the container metadata (`average_rate` or `base_rate`).
2. If the source FPS differs from `target_fps` by more than 1%, frames are resampled (dropped) to match `target_fps`.
3. If the source and target FPS are within 1% of each other (e.g., 23.976 vs 24), no resampling is done — this avoids spurious frame drops from NTSC rounding (23.976, 29.97, 59.94, etc.).
4. If audio is present (`--ltx2_mode av`), the audio waveform is automatically time-stretched (pitch-preserving) to match the resampled video duration.

#### Common Scenarios

**Default — no FPS config needed (recommended for most users):**
```toml
[[datasets]]
video_directory = "videos"
target_frames = [1, 17, 33, 49]
# source_fps: auto-detected per video
# target_fps: defaults to 24
```
A 60fps video produces 240 frames per 10 seconds (not 600). A 30fps video produces 240 frames per 10 seconds (not 300). A 24fps video is passed through as-is. Mixed-FPS datasets work correctly — each video is resampled independently.

**Training at a non-standard frame rate (e.g., 60fps):**
```toml
[[datasets]]
video_directory = "videos_60fps"
target_frames = [1, 17, 33, 49]
target_fps = 60
```
Set `target_fps` to the desired training rate. Videos at 60fps (or 59.94fps) pass through without resampling. Videos at other frame rates are resampled to 60fps. Note: LTX-2 was trained on 24fps content, so non-24fps training is experimental.

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
Resampling my_video.mp4: 60.00 FPS -> 24.00 FPS
```
If you see **no** "Resampling" line for a video, it means source and target FPS matched (within 1%) and all frames were kept as-is. If you see unexpected frame counts in your cached latents, check these log lines first.

#### Quick Reference

| Your situation | What to set | What happens |
|---|---|---|
| Mixed FPS dataset, want 24fps training | Nothing (defaults work) | Each video auto-detected, resampled to 24fps |
| All videos are 24fps | Nothing | Auto-detected as 24fps, no resampling (within 1%) |
| All videos are 60fps, want 60fps training | `target_fps = 60` | Auto-detected as 60fps, no resampling |
| All videos are 60fps, want 24fps training | Nothing | Auto-detected as 60fps, resampled to 24fps |
| VFR videos with wrong detection | `source_fps = 30` (your actual FPS) | Overrides auto-detection |
| Image directory | Nothing | No FPS concept, all images loaded |

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
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Missing cache keys during training | Caching incomplete | Run both `ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py` |
| Missing `*_ltx2_audio.safetensors` | Audio caching skipped | Re-run latent caching with `--ltx2_mode av` |
| Gemma connector weights missing | Incorrect checkpoint | Ensure `--ltx2_checkpoint` (or `--ltx2_text_encoder_checkpoint`) contains Gemma connector weights |
| Gemma OOM | Model too large | Use `--gemma_load_in_8bit` or `--gemma_load_in_4bit` with `--device cuda` |
| Audio caching fails | torchaudio missing | Install torchaudio before running `ltx2_cache_latents.py` |
| Sampling OOM | VAE decode too large | Enable `--sample_tiled_vae` or reduce `--sample_vae_temporal_tile_size` |
| Crash with block swap (esp. RTX 5090) | `--use_pinned_memory_for_block_swap` bug | Remove `--use_pinned_memory_for_block_swap` from training arguments |
| Wrong frame count in cached latents | Auto-detected FPS incorrect (e.g., VFR video) | Set `source_fps` explicitly in TOML config to override auto-detection |
| Too few frames from high-FPS video | FPS resampling working correctly (e.g., 60fps→24fps = 40% of frames) | This is expected behavior. Set `target_fps = 60` if you want to keep all frames |
| Audio/video out of sync after caching | Source FPS mismatch causing wrong time-stretch | Check "Auto-detected source FPS" log line; set `source_fps` explicitly if wrong |

---

## 4. Slider LoRA Training

Slider LoRAs learn a controllable direction in model output space (e.g., "detailed" vs "blurry"). At inference, you scale the LoRA multiplier to control the effect strength and direction: `+1.0` enhances, `-1.0` erases, `0.0` is the base model, and values like `+2.0` or `-0.5` work too.

**Script:** `ltx2_train_slider.py`

Two modes are available:

| Mode | Input | Use Case |
|------|-------|----------|
| `text` | Prompt pairs only (no dataset) | Quick concept sliders from text descriptions |
| `reference` | Pre-cached latent pairs | Precise control from paired positive/negative images |

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
- `weight`: Per-target loss weight. Useful when training multiple directions simultaneously.
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

Learns a slider direction from paired positive/negative images (or videos). Requires pre-cached latents.

#### Step 1: Prepare Paired Data

Create two directories with matching filenames — one with positive-attribute images, one with negative:

```
positive_images/        negative_images/
  img_001.png             img_001.png      # same subject, different attribute
  img_002.png             img_002.png
  img_003.png             img_003.png
```

Each positive image must have a corresponding negative image with the same filename. The images should depict the same subject but differ in the target attribute (e.g., smiling vs neutral face, detailed vs blurry).

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
pos_cache_dir = "path/to/positive/cache"
neg_cache_dir = "path/to/negative/cache"
text_cache_dir = "path/to/positive/cache"   # can be same as pos (text is shared)
sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

- `pos_cache_dir`: Directory containing cached positive latents (output of `ltx2_cache_latents.py`)
- `neg_cache_dir`: Directory containing cached negative latents
- `text_cache_dir`: Directory containing cached text encoder outputs. Defaults to `pos_cache_dir` if omitted.

Pairs are matched by filename: for each `{name}_{W}x{H}_ltx2_v1.safetensors` in the positive directory, a matching file must exist in the negative directory. Unmatched files are skipped with a warning.

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

### Slider Tips

- **Start small**: `--network_dim 8` or `16` with `--max_train_steps 200-500` is usually sufficient.
- **Monitor loss**: Loss should decrease steadily. If it diverges, reduce `--learning_rate`.
- **Preview samples**: Add `--sample_prompts sampling_prompts.txt --sample_every_n_steps 50` to generate previews at each slider strength during training. Requires `--gemma_root` for text encoding.
- **Guidance strength**: For text-only mode, `1.0` is a safe default. Values of `2.0-3.0` produce stronger but potentially less stable sliders.
- **Multiple targets**: Text-only mode supports multiple `[[targets]]` blocks. Each step randomly selects one target, so all directions get trained evenly.
- **Inference**: Use the trained LoRA with any multiplier value. Positive multipliers enhance the positive attribute, negative multipliers enhance the negative attribute. Values beyond `[-1, +1]` extrapolate the effect.

---

## Useful Links

- [Installation Guide](https://github.com/AkaneTendo25/musubi-tuner/discussions/19) — Setup instructions, dependencies, flash-attn, troubleshooting
- [Optimizers Guide](https://github.com/AkaneTendo25/musubi-tuner/discussions/21) — Optimizer comparison, recommended settings, memory usage tips

