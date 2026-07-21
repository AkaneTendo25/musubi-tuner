# MagiHuman

This page describes **base-model MagiHuman LoRA training**.

Covered:
- Base `daVinci-MagiHuman` LoRA training
- Cached text embeddings
- Cached video/audio/image latents
- Single-GPU training with FP8 + block swap

Not covered:
- SR-stage training
- Standalone inference workflow

## 1. Download the weights

Use these sources:

- **Base DiT**: `base_bf16.safetensors`
  - https://huggingface.co/AkaneTendo25/daVinci-MagiHuman/blob/main/base_bf16.safetensors
- **T5-Gemma encoder-only**:
  - https://huggingface.co/PhatcatDK/t5gemma-9b-2b-ul2-encoder-only/tree/main
- **Wan 2.2 VAE**:
  - https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/Wan2.2_VAE.pth
- **Stable Audio Open 1.0**:
  - https://huggingface.co/audo/stable-audio-open-1.0

Example local layout:

```text
path\to\daVinci-MagiHuman\
├── base_bf16.safetensors              
├── t5gemma\
├── wan22_vae\
│   └── Wan2.2_VAE.pth
└── sao\
    ├── model.safetensors
    └── model_config.json
```

Notes:
- `t5gemma` should be a local Hugging Face-style folder.
- `sao` must contain both `model.safetensors` and `model_config.json`.
- `ffmpeg` must be available in `PATH`.

If you start from the original sharded checkpoint instead of the single-file
`base_bf16.safetensors`, convert it with
[`magihuman_convert_checkpoint.py`](../magihuman_convert_checkpoint.py):

```bat
python magihuman_convert_checkpoint.py ^
  --input path/to/magihuman_checkpoint_dir ^
  --output path/to/daVinci-MagiHuman/base_bf16.safetensors ^
  --dtype bf16
```

## 2. Prepare the dataset config

Create a dataset config, for example `magihuman_lora_dataset.toml`.

Set:
- `video_directory`
- `cache_directory`

Captions are expected as `.txt` files next to each video.

Example:

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
num_repeats = 1
enable_bucket = true

[[datasets]]
video_directory = "path/to/your/dataset/videos"
cache_directory = "path/to/your/dataset/videos/cache"
target_frames = [33]
frame_sample = 1
frame_extraction = "head"
```

## 3. Cache text embeddings

Use [`magihuman_cache_text_encoder_outputs.py`](../magihuman_cache_text_encoder_outputs.py):

```bat
python magihuman_cache_text_encoder_outputs.py ^
  --dataset_config magihuman_lora_dataset.toml ^
  --text_encoder path/to/daVinci-MagiHuman/t5gemma ^
  --device cuda ^
  --weight_dtype bfloat16 ^
  --batch_size 1 ^
  --num_workers 1 ^
  --skip_existing ^
  --t5gemma_load_in_8bit
```

## 4. Cache latents

Use [`magihuman_cache_latents.py`](../magihuman_cache_latents.py):

```bat
python magihuman_cache_latents.py ^
  --dataset_config magihuman_lora_dataset.toml ^
  --vae path/to/daVinci-MagiHuman/wan22_vae/Wan2.2_VAE.pth ^
  --audio_model path/to/daVinci-MagiHuman/sao ^
  --device cuda ^
  --vae_dtype bf16 ^
  --batch_size 1 ^
  --num_workers 1 ^
  --skip_existing
```

## 5. Start training

Example training settings:
- base checkpoint from `path/to/daVinci-MagiHuman/base_bf16.safetensors`
- `--dit_dtype bfloat16`
- `--gradient_checkpointing`
- `--blocks_to_swap 10`
- `--fp8_base --fp8_scaled`

```bat
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 magihuman_train_network.py ^
  --optimizer_type AdamW8bit ^
  --dataset_config magihuman_lora_dataset.toml ^
  --dit path/to/daVinci-MagiHuman/base_bf16.safetensors ^
  --dit_dtype bfloat16 ^
  --magihuman_fp8_quant_device cuda ^
  --magihuman_frame_receptive_field -1 ^
  --sdpa ^
  --network_module networks.lora_magihuman ^
  --network_dim 32 ^
  --network_alpha 32 ^
  --learning_rate 1e-4 ^
  --max_data_loader_n_workers 0 ^
  --max_train_steps 5000 ^
  --save_every_n_steps 500 ^
  --output_dir output/magihuman_lora ^
  --output_name magihuman_lora ^
  --gradient_checkpointing ^
  --blocks_to_swap 10 ^
  --fp8_base ^
  --fp8_scaled
```

## 6. Run order

1. Edit `magihuman_lora_dataset.toml`
2. Run [`magihuman_cache_text_encoder_outputs.py`](../magihuman_cache_text_encoder_outputs.py)
3. Run [`magihuman_cache_latents.py`](../magihuman_cache_latents.py)
4. Run [`magihuman_train_network.py`](../magihuman_train_network.py)

## 7. Notes

- The example command uses `batch_size = 1`.
- If VRAM is insufficient, reducing resolution reduces token count and activation memory.
- This training path targets the **base model**, not the SR stack.
- `--magihuman_t_patch_size`, `--magihuman_patch_size`, `--magihuman_spatial_rope_interpolation`
  and `--magihuman_coords_style` default to the base-model values and can be omitted.
  `--magihuman_text_offset` applies only with `--magihuman_coords_style v1`; setting it to a
  non-default value under `v2` is rejected.
- Base training does not enable local-attention layers, so `frame_receptive_field` is forced
  to `-1`; passing `--magihuman_frame_receptive_field -1` explicitly just avoids the warning.
- For training-time samples, add `--sample_prompts ...` plus `--vae`, `--text_encoder`, and `--audio_model`.
