# LTX-2

This guide details the three-step process for training LTX-2 LoRA models:
1. **Caching Latents** (Video/Image + Audio)
2. **Caching Text Encoder Outputs** (Prompts)
3. **Training**

## Supported Dataset Types

| Mode | Dataset Type | Notes |
|------|--------------|-------|
| `video` | Images | Treated as 1-frame samples (`F=1`) |
| `video` | Videos | Standard video training |
| `av` | Videos with audio | Audio extracted from video or separate `.wav` files |
| `audio` | Audio only | Video latents are dummy placeholders; only audio loss is used |

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
  --ltx_mode av ^
  --ltx2_audio_source video
```

### Key Arguments
- `--ltx_mode av`: Enables Audio-Video processing. Caches both `*_ltx2.safetensors` (video) and `*_ltx2_audio.safetensors` (audio) latents.
- `--ltx2_audio_source video`: Extracts audio directly from the video file. Use `wav` if you have separate audio files.
- `--vae_dtype bf16`: Recommended precision for VAE latents.

### Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2.safetensors` | Video latents: `latents_{F}x{H}x{W}_{dtype}` |
| `*_ltx2_audio.safetensors` | Audio latents: `audio_latents_{T}x{mel_bins}x{C}_{dtype}`, `audio_lengths_int32` |

### Memory Optimization for Caching
If you encounter Out-Of-Memory (OOM) errors during caching (especially with higher resolutions like 1080p), use VAE chunking and spatial tiling:

```bash
python ltx2_cache_latents.py ^
  ...
  --vae_chunk_size 16 ^
  --vae_spatial_tile_size 512 ^
  --vae_spatial_tile_overlap 64
```

- `--vae_chunk_size`: Processes video in temporal chunks (e.g., 16 or 32 frames at a time). Default: `None` (all frames).
- `--vae_spatial_tile_size`: Processes video in spatial tiles (e.g., 512x512 pixels). Default: `None` (full image).
- `--vae_spatial_tile_overlap`: Overlap between spatial tiles (pixels). Default: `64`.

---

## 2. Caching Text Encoder Outputs

This step pre-computes text embeddings using the Gemma-3-12B model.

**Script:** `ltx2_cache_text_encoder_outputs.py`

### Example Command
```bash
python ltx2_cache_text_encoder_outputs.py ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --gemma_root /path/to/gemma-3-12b-it ^
  --gemma_load_in_8bit ^
  --device cuda ^
  --mixed_precision bf16 ^
  --ltx_mode av ^
  --batch_size 1
```

### Key Arguments
- `--gemma_root`: Path to the local Gemma-3-12b-it model folder (Hugging Face format).
- `--gemma_load_in_8bit`: Loads Gemma in 8-bit quantization.
- `--gemma_load_in_4bit`: Loads Gemma in 4-bit quantization.
- `--gemma_safetensors`: Optional. Load Gemma weights from a single `.safetensors` file (tokenizer/config still come from `--gemma_root`).
- `--ltx_mode av`: MUST match the mode used in latent caching. Concatenates video and audio prompt embeddings.

### Output Files

| File Pattern | Contents |
|--------------|----------|
| `*_ltx2_te.safetensors` | `video_prompt_embeds_{dtype}`, `audio_prompt_embeds_{dtype}` (av only), `prompt_attention_mask` |

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
  --gemma_root /path/to/gemma-3-12b-it ^
  --separate_audio_buckets ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --ltx_mode av ^
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
- `--fp8_base`, `--fp8_scaled`: Casts base model weights to FP8. Essential for training on 24GB VRAM.
- `--blocks_to_swap X`: Offloads X transformer blocks to CPU (max 47 for 48-block model). Higher values save more VRAM but increase CPU↔GPU overhead.
- `--gradient_checkpointing`: Reduces VRAM by recomputing activations during backward pass. **Required** for blockwise checkpointing.
- `--gradient_checkpointing_cpu_offload`: Offloads activations to CPU during gradient checkpointing recomputation (activation-only offload).
- `--blockwise_checkpointing`: Enables block-level gradient checkpointing with optional weight offloading for checkpointed blocks. Requires `--gradient_checkpointing`.
- `--blocks_to_checkpoint N`: Number of blocks to checkpoint. `-1` = all, `0` = none, `N` = last N blocks.
- `--fp8_offload_upcast`: **Optional**. Allow FP8 weights to be offloaded to CPU by upcasting to bf16 (fallback to fp16) and restoring on GPU. This can reduce VRAM but may affect training stability.
- `--fp8_offload_restore_bf16`: **Optional**. When used with `--fp8_offload_upcast`, keep restored weights in bf16 on GPU to avoid FP8 round‑trip noise (uses more VRAM).
- `--fp8_offload_restore_stochastic`: **Experimental**. Add stochastic rounding noise before restoring FP8 weights (may increase noise; use for testing).
- `--use_pinned_memory_for_block_swap`: Uses pinned memory for faster CPU↔GPU block transfers.
- `--swap_norms`: Also swaps RMSNorm/LayerNorm weights to CPU. Extra memory savings.

#### Aggressive VRAM Optimization (8-16GB GPUs)

For maximum VRAM savings on 8-16GB GPUs, use this combination of flags:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py ^
  --mixed_precision bf16 ^
  --dataset_config dataset.toml ^
  --ltx2_checkpoint /path/to/ltx-2.safetensors ^
  --ltx_mode av ^
  --gemma_load_in_8bit ^
  --gemma_root /path/to/gemma-3-12b-it ^
  --fp8_base ^
  --fp8_scaled ^
  --blocks_to_swap 47 ^
  --use_pinned_memory_for_block_swap ^
  --gradient_checkpointing ^
  --gradient_checkpointing_cpu_offload ^
  --blockwise_checkpointing ^
  --blocks_to_checkpoint 47 ^
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
- Use `--blockwise_checkpointing` with `--gradient_checkpointing` (activation recompute + per-block weight offload)
- Use `--blocks_to_checkpoint -1` if you are not using block swap and want to checkpoint all blocks
- Use smaller LoRA rank (`--network_dim 16` instead of 32)
- Use smaller training resolutions (e.g., 512x320)
- Reduce `--sample_vae_temporal_tile_size` to 24 or lower
- Use `--use_pinned_memory_for_block_swap` - faster transfers

#### Audio-Video Support
- `--ltx_mode av`: Enables joint Video+Audio training logic.
- `--separate_audio_buckets`: Keeps audio and non-audio items in separate batches (reduces VRAM for image/video-only batches).

#### Loss Weighting
- `--video_loss_weight`: Weight for video loss (default: 1.0).
- `--audio_loss_weight`: Weight for audio loss in AV mode (default: 1.0).

#### Timestep Sampling
- `--timestep_sampling shifted_logit_normal`: **(Recommended)** Official LTX-2 method. Uses a shifted logit-normal distribution where the shift is computed based on sequence length (frames × height × width). This is the default for LTX-2.
- `--timestep_sampling uniform`: Simple uniform sampling from [0, 1]. Alternative if you want simpler behavior.
- `--logit_std`: Standard deviation for the logit-normal distribution (default: 1.0). Only used with `shifted_logit_normal`.
- `--min_timestep` / `--max_timestep`: Optional timestep range constraints.

**Note:** The `shifted_logit_normal` mode matches the official Lightricks LTX-2 trainer exactly. The shift is linearly interpolated from 0.95 (at 1024 tokens) to 2.05 (at 4096 tokens) based on sequence length.

#### LoRA Targets
By default, LoRA targets attention layers only (`to_q`, `to_k`, `to_v`, `to_out.0`), matching the official LTX-2 trainer. This applies to all attention types: self-attention, cross-attention, and cross-modal attention (in AV mode).

To target additional layers (e.g., FFN), use `--network_args`:
```bash
--network_args "include_patterns=['.*\.to_k$','.*\.to_q$','.*\.to_v$','.*\.to_out\.0$','.*\.ff\.net\.0\.proj$','.*\.ff\.net\.2$']"
```

#### Sampling with Tiled VAE
- `--sample_tiled_vae`: **Required** for high resolution previews.
- `--sample_vae_tile_size 512`: Spatial tile size.
- `--sample_vae_tile_overlap 64`: Spatial overlap (pixels).
- `--sample_vae_temporal_tile_size 48`: Number of frames to decode at once. Set to 33 or lower if OOM.
- `--sample_vae_temporal_tile_overlap 8`: Temporal overlap (frames).
- `--sample_merge_audio`: Merges generated audio into the `.mp4`.

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
| Missing `*_ltx2_audio.safetensors` | Audio caching skipped | Re-run latent caching with `--ltx_mode av` |
| `embeddings_connector` uninitialized | Incorrect checkpoint | Ensure `--ltx2_checkpoint` contains `model.diffusion_model.video_embeddings_connector.*` keys |
| Gemma OOM | Model too large | Use `--gemma_load_in_8bit` or `--gemma_load_in_4bit` |
| Sampling OOM | VAE decode too large | Enable `--sample_tiled_vae` and reduce `--sample_vae_temporal_tile_size` |

