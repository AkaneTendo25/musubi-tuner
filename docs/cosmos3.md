# Cosmos3-Nano LoRA

Cosmos3-Nano LoRA training and inference.

## Weights

Official weights:

- Hugging Face: `nvidia/Cosmos3-Nano`
- Model page: <https://huggingface.co/nvidia/Cosmos3-Nano>
- Reference code: <https://github.com/nvidia/cosmos>

Example local layout:

```text
G:\storage\cosmos3-nano
├── scheduler
├── sound_tokenizer
├── text_tokenizer
├── transformer
└── vae
```

Download example:

```powershell
huggingface-cli login
huggingface-cli download nvidia/Cosmos3-Nano --local-dir G:\storage\cosmos3-nano
```

The scripts accept either a local path or a Hugging Face repo id. Local paths are recommended for training.

## Dataset

Use a normal Musubi video dataset TOML. The Cosmos3 cache and training scripts set the dataset architecture internally.

For video+audio training, source files must contain an audio stream. Audio is cropped with the same frame window as the video crop.

## Cache Text Tokens

Cosmos3-Nano does not use a separate text encoder. This cache stores Qwen token IDs.

```powershell
python cosmos3_cache_text_encoder_outputs.py `
  --dataset_config path\to\dataset.toml `
  --model G:\storage\cosmos3-nano `
  --tokenizer_subfolder text_tokenizer `
  --fps 24
```

Rerun this cache after changing prompt template flags or captions.

## Cache Latents

Video-only:

```powershell
python cosmos3_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae G:\storage\cosmos3-nano `
  --vae_subfolder vae `
  --vae_dtype bfloat16
```

Video+audio:

```powershell
python cosmos3_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae G:\storage\cosmos3-nano `
  --vae_subfolder vae `
  --vae_dtype bfloat16 `
  --cache_audio `
  --sound_tokenizer G:\storage\cosmos3-nano `
  --sound_tokenizer_subfolder sound_tokenizer `
  --sound_sample_rate 48000 `
  --sound_channels 2 `
  --sound_latent_fps 25
```

If video-only caches already exist, rerun latent caching without `--skip_existing` to add `sound_latents_*`.

## Train

Video+audio LoRA, scaled FP8 base, gradient checkpointing, block swap, samples/checkpoints every 300 steps:

```powershell
python -m accelerate.commands.launch cosmos3_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit G:\storage\cosmos3-nano `
  --vae G:\storage\cosmos3-nano `
  --transformer_subfolder transformer `
  --vae_subfolder vae `
  --tokenizer_subfolder text_tokenizer `
  --sound_tokenizer G:\storage\cosmos3-nano `
  --sound_tokenizer_subfolder sound_tokenizer `
  --sound_dtype bfloat16 `
  --mixed_precision bf16 `
  --sdpa `
  --fp8_base --fp8_scaled `
  --gradient_checkpointing `
  --blocks_to_swap 12 `
  --use_pinned_memory_for_block_swap `
  --audio `
  --sound_latent_fps 25 `
  --sound_loss_weight 1.0 `
  --fps 24 `
  --vae_dtype bfloat16 `
  --vae_scale_factor_temporal 4 `
  --network_dim 32 `
  --network_alpha 32 `
  --network_module musubi_tuner.networks.lora_cosmos3 `
  --learning_rate 1e-4 `
  --optimizer_type adamw `
  --max_train_steps 1000 `
  --save_every_n_steps 300 `
  --save_state_on_train_end `
  --sample_prompts path\to\sample_prompts.txt `
  --sample_every_n_steps 300 `
  --offload_dit_during_sampling `
  --max_data_loader_n_workers 1 `
  --output_dir output\cosmos3_lora `
  --output_name cosmos3_lora `
  --timestep_sampling shift `
  --discrete_flow_shift 10.0
```

Notes:

- `--audio` requires latent caches with `sound_latents_*`.
- `--fp8_base --fp8_scaled` quantizes base weights; LoRA weights remain trainable.
- `--blocks_to_swap 12` is the tested 24 GB setting for 512x288 / 49-frame smoke training.
- `--offload_dit_during_sampling` avoids holding DiT, VAE, and AVAE on GPU at the same time during previews.

## Inference

Text-to-video:

```powershell
python cosmos3_generate_video.py `
  --dit G:\storage\cosmos3-nano `
  --vae G:\storage\cosmos3-nano `
  --prompt "a person walking through a quiet town at dusk" `
  --negative_prompt "blurry, distorted, low quality" `
  --video_size 512 288 `
  --video_length 49 `
  --infer_steps 35 `
  --guidance_scale 6 `
  --flow_shift 10.0 `
  --fp8_base --fp8_scaled `
  --blocks_to_swap 12 `
  --offload_dit_during_sampling `
  --save_path output\cosmos3_sample.mp4
```

With LoRA:

```powershell
python cosmos3_generate_video.py `
  --dit G:\storage\cosmos3-nano `
  --vae G:\storage\cosmos3-nano `
  --lora_weight output\cosmos3_lora\cosmos3_lora.safetensors `
  --prompt "a person walking through a quiet town at dusk" `
  --negative_prompt "blurry, distorted, low quality" `
  --video_size 512 288 `
  --video_length 49 `
  --infer_steps 35 `
  --guidance_scale 6 `
  --flow_shift 10.0 `
  --fp8_base --fp8_scaled `
  --blocks_to_swap 12 `
  --offload_dit_during_sampling `
  --save_path output\cosmos3_lora_sample.mp4
```

Add `--audio --sound_tokenizer G:\storage\cosmos3-nano --sound_tokenizer_subfolder sound_tokenizer` to decode AVAE audio and write a WAV next to the video.
