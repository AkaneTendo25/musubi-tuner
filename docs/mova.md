# MOVA LoRA Training

MOVA is now integrated into the standard musubi cache/train flow instead of a standalone cache-dir trainer.

Available entrypoints:

- `python mova_cache_text_encoder_outputs.py`
- `python mova_cache_latents.py`
- `python mova_train_network.py`
- `python mova_generate_video.py`

The MOVA LoRA implementation lives in `musubi_tuner.networks.lora_mova` and targets the full MOVA bundle:

- visual tower `video_dit`
- optional alternate visual tower `video_dit_2`
- audio tower `audio_dit`
- dual-tower bridge `dual_tower_bridge`

## Dataset Config

Use the normal musubi dataset config format.

Directory-based example:

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
video_directory = "/data/mova/videos"
audio_directory = "/data/mova/audio"
cache_directory = "/data/mova/cache"
target_frames = [81]
frame_extraction = "slide"
frame_stride = 8
source_fps = 30.0
```

JSONL-based example:

```toml
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true

[[datasets]]
video_jsonl_file = "/data/mova/metadata.jsonl"
cache_directory = "/data/mova/cache"
target_frames = [81]
frame_extraction = "head"
```

JSONL lines may include `audio_path`:

```json
{"video_path":"/data/mova/video1.mp4","audio_path":"/data/mova/audio1.wav","caption":"a drum solo in a warehouse"}
```

If `audio_directory` is set, MOVA matches audio files to videos by basename.

## Cache Text Encoder Outputs

```bash
python mova_cache_text_encoder_outputs.py \
  --dataset_config ./mova_dataset.toml \
  --text_encoder /models/MOVA \
  --text_encoder_subfolder text_encoder \
  --tokenizer /models/MOVA \
  --tokenizer_subfolder text_encoder \
  --text_encoder_dtype bfloat16
```

The cache script supports encoder-only and encoder-decoder Hugging Face text encoders, so T5/UMT5-style checkpoints work.

## Cache Latents

```bash
python mova_cache_latents.py \
  --dataset_config ./mova_dataset.toml \
  --vae /models/MOVA \
  --video_vae_subfolder video_vae \
  --audio_vae /models/MOVA \
  --audio_vae_subfolder audio_vae \
  --audio_vae_type dac \
  --audio_vae_model_spec path/to/dac_vae.py:DAC \
  --vae_dtype bfloat16
```

Notes:

- `--audio_vae` defaults to `--vae` if omitted.
- If your DAC class is not importable from the environment, pass `--audio_vae_model_spec`.
- If `audio_directory` is not set, the script tries to read audio from the source video itself.

## Train LoRA

```bash
python mova_train_network.py \
  --dataset_config ./mova_dataset.toml \
  --pretrained_model_path /models/MOVA \
  --visual_subfolder video_dit \
  --alternate_visual_subfolder video_dit_2 \
  --audio_subfolder audio_dit \
  --bridge_subfolder dual_tower_bridge \
  --output_dir ./output/mova \
  --output_name mova_lora \
  --learning_rate 1e-4 \
  --network_dim 32 \
  --network_alpha 32 \
  --lora_scope official \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --blocks_to_swap 8 \
  --fp8_scaled \
  --max_train_steps 1000 \
  --sdpa
```

Supported training features now follow the normal musubi path:

- dataset blueprint / `dataset_config`
- cached text encoder outputs
- cached latents
- standard save/resume/state metadata flow
- `--gradient_checkpointing`
- `--blocks_to_swap`
- `--fp8_scaled`
- sample generation via `--sample_prompts`

`--blocks_to_swap` now covers the bridge conditioners as well as the visual/audio towers.

For dual-visual-tower training, `--offload_inactive_dit` is also available as an alternative memory-saving mode, but it cannot be combined with `--blocks_to_swap`.

## LoRA Scope

`--lora_scope official` is the default and targets attention projections across the MOVA bundle:

- visual/audio `DiTBlock` attention `q/k/v/o`
- bridge `ConditionalCrossAttentionBlock` attention `q/k/v/o`

Available scopes:

- `official`
- `attention_plus_image`
- `all_linear`

## Sample Prompts

Training-time sampling is available through the normal musubi `--sample_prompts` path.

Example:

```bash
python mova_train_network.py \
  ... \
  --vae /models/MOVA \
  --video_vae_subfolder video_vae \
  --text_encoder /models/MOVA \
  --text_encoder_subfolder text_encoder \
  --tokenizer /models/MOVA \
  --tokenizer_subfolder text_encoder \
  --sample_prompts ./sample_prompts.txt \
  --sample_every_n_steps 100
```

Prompt file example:

```text
a cinematic drummer on a rooftop at sunset --w 960 --h 544 --f 81 --s 20 --d 42
```

If you include `--i /path/to/image.png` in a prompt line, the sample hook encodes that image as the MOVA conditioning latent. Without `--i`, sample previews use zero conditioning.

Training-time samples now decode audio and mux it into the preview `.mp4`.

## Standalone Inference

```bash
python mova_generate_video.py \
  --pretrained_model_path /models/MOVA \
  --vae /models/MOVA \
  --video_vae_subfolder video_vae \
  --text_encoder /models/MOVA \
  --text_encoder_subfolder text_encoder \
  --tokenizer /models/MOVA \
  --tokenizer_subfolder text_encoder \
  --alternate_visual_subfolder video_dit_2 \
  --save_path ./output/mova_gen \
  --prompt "a cinematic drummer on a rooftop at sunset" \
  --video_size 544 960 \
  --video_length 81 \
  --fps 16 \
  --infer_steps 20 \
  --guidance_scale 5.0 \
  --save_wav
```

This script writes an `.mp4` with muxed decoded audio, and optionally a `.wav` sidecar.
