# LTX-2

Status: LTX-2 support is work in progress and may be incomplete or unstable.

This guide details the three-step process for training LTX-2 LoRA models:
1. **Caching Latents** (Video/Image + Audio)
2. **Caching Text Encoder Outputs** (Prompts)
3. **Training**

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

