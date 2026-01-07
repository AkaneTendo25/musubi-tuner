# LTX-2 (Video / Audio-Video) Training

This document describes the **LTX-2** training pipeline implemented in this repository.

- Trainer entry point: `ltx2_train_network.py` (repo root)
- Implementation: `src/musubi_tuner/ltx2_train_network.py`
- Network module (default): `musubi_tuner.networks.lora_ltx2`

## Supported modalities

The modality is selected via `--ltx_mode`:

- `video`
  - Video-only training
- `av`
  - Audio-video training
  - Requires audio latent sidecar cache files
- `audio`
  - Reserved for future audio-only training (planned text-to-music / music diffusion use)

Use `--ltx_mode av` for audio-video.

## Supported dataset types

LTX-2 uses musubi-tuner's standard dataset config system and supports:

- **Image datasets**
  - Supported for `--ltx_mode video`.
  - Images are treated as 1-frame samples (i.e., `F=1` in the latent cache).
- **Video datasets**
  - Supported for `--ltx_mode video` and `--ltx_mode av`.
  - For `--ltx_mode av`, each item must have an audio source (either embedded in the video file or a separate audio file depending on your caching flags).

## Required caching workflow

Training expects **cached** data on disk (latents and text embeddings) produced by the dedicated caching scripts.

### 1) Cache video latents (and audio latents for AV)

Script: `ltx2_cache_latents.py`

This produces per-item video latent cache files containing keys like:

- `latents_{F}x{H}x{W}_{dtype}`

and writes them to files like:

- `*_ltx2.safetensors`

If you run with `--ltx_mode av`, the same script also produces an **audio sidecar** file next to each video latent cache:

- `*_ltx2_audio.safetensors`

with keys:

- `audio_latents_{T}x{mel_bins}x{C}_{dtype}`
- `audio_lengths_int32`

Notes:

- Audio decoding supports PyAV-based fallback on Windows.
- Multi-channel audio is downmixed to stereo before encoding.
- Mel-spectrogram computation runs in float32 for stability; outputs are cast to the configured audio dtype before the encoder.

Audio latents are only cached for **video dataset items**.

### 2) Cache text encoder outputs (Gemma)

Script: `ltx2_cache_text_encoder_outputs.py`

It writes a per-item file:

- `*_ltx2_te.safetensors`

with keys:

- `video_prompt_embeds_{dtype}`
- `audio_prompt_embeds_{dtype}` (only if `--ltx_mode av`)
- `prompt_attention_mask`

For convenience/compatibility it also writes:

- `text_{dtype}`
- `text_mask`

where `text` is either:

- video-only: `video_prompt_embeds`
- AV: `cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)`

Gemma notes:

- `--gemma_root` points to the **Gemma weights/tokenizer**.
- `--ltx2_checkpoint` provides the **connector/projection weights** (e.g. video/audio embeddings connector).
- Optional: `--ltx2_text_encoder_checkpoint` can be used if your connector weights live in a different `.safetensors` file.

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
- For AV caching, audio is resolved either from the video file itself or from separate audio files depending on your `ltx2_cache_latents.py` flags.

### Typical cache directory layout

The dataset config usually specifies a `cache_directory`. After caching, you should expect files like:

```
cache_directory/
  000001_1024x0576_ltx2.safetensors          # video latents
  000001_ltx2_te.safetensors                 # text encoder outputs
  000001_ltx2_audio.safetensors              # audio latents (AV only)
```

The dataset loader merges the video latent cache, optional audio sidecar, and text cache into one training batch.

### How caches are merged into a batch

The dataset batch manager merges multiple `.safetensors` sources per item:

1. video latent cache (`item_info.latent_cache_path`, `*_ltx2.safetensors`)
2. optional audio latent cache sidecar (`item_info.audio_latent_cache_path`, `*_ltx2_audio.safetensors`)
3. text encoder output cache (`item_info.text_encoder_output_cache_path`, `*_ltx2_te.safetensors`)

The merged batch contains (at minimum):

- `batch["latents"]`
- `batch["conditions"]` with:
  - `video_prompt_embeds`
  - `audio_prompt_embeds` (AV only)
  - `prompt_attention_mask` (optional)

If only `text/text_mask` are present, the dataset will attempt to derive `video_prompt_embeds` (and `audio_prompt_embeds` by splitting the last dim when possible).

## Training command (canonical)

Use `accelerate` for training.

Important notes:

- For LTX-2, the trainer uses `--ltx2_checkpoint` for both the DiT and VAE internally.
  If you pass `--dit` or `--vae`, they are ignored.
- `--gemma_root` is only required if you use `--sample_prompts` (preview sampling). Training itself uses cached embeddings.

### Example: video-only (T2V)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --ltx_mode video \
  --network_dim 16 \
  --network_alpha 16
```

### Example: audio-video (AV)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --ltx_mode av \
  --audio_loss_weight 1.0 \
  --network_dim 16 \
  --network_alpha 16
```

## End-to-end example workflow

### Step 1: cache video latents

```bash
python ltx2_cache_latents.py \
  --dataset_config /path/to/dataset.toml \
  --vae /path/to/ltx2_checkpoint.safetensors \
  --vae_dtype bf16
```

### Step 2: cache text encoder outputs

Video-only (T2V):

```bash
python ltx2_cache_text_encoder_outputs.py \
  --dataset_config /path/to/dataset.toml \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --mixed_precision bf16
```

Audio-video (T2AV): cache concatenated `[video_ctx, audio_ctx]` embeddings:

```bash
python ltx2_cache_text_encoder_outputs.py \
  --dataset_config /path/to/dataset.toml \
  --ltx_mode av \
  --ltx2_checkpoint /path/to/ltx2_checkpoint.safetensors \
  --gemma_root /path/to/local/gemma_root \
  --mixed_precision bf16
```

### Step 3: cache audio latents (T2AV only)

Audio latents are produced by `ltx2_cache_latents.py` when `--ltx_mode av` is used.

### Step 4: train

Use the training commands above (T2V or T2AV).

## Preview sampling (`--sample_prompts`)

If `--sample_prompts` is provided, the trainer will:

- Load the Gemma text encoder using `--ltx2_checkpoint` + `--gemma_root`
- Encode prompt(s)
- Run a FlowMatchDiscreteScheduler denoising loop
- Decode latents with the VAE and save preview output

To match the reference trainer behavior, Gemma is unloaded after prompt embeddings are computed to free VRAM.

## Loss weighting

LTX-2 supports the following loss weights:

- `--video_loss_weight`
- `--audio_loss_weight` (AV only)

## Known limitations

- `--ltx_mode audio`
  - Not implemented.
- AV inference previews
  - The current preview path is video-only; audio preview generation is not implemented.

## Troubleshooting

- If training fails with missing cache keys:
  - Ensure you ran both `ltx2_cache_latents.py` and `ltx2_cache_text_encoder_outputs.py`.
  - For AV, ensure `*_ltx2_audio.safetensors` exist.
- If text encoder caching fails with uninitialized `embeddings_connector.*`:
  - Ensure your `--ltx2_checkpoint` actually contains:
    - `model.diffusion_model.video_embeddings_connector.*` (and audio connector keys for AV)
    - `text_embedding_projection.*`
  - Ensure `--gemma_root` points to the local Gemma model directory.
