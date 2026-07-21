# Cosmos3-Nano LoRA

Cosmos3-Nano LoRA training and inference.

## Weights

Official weights:

- Hugging Face: `nvidia/Cosmos3-Nano`
- Model page: <https://huggingface.co/nvidia/Cosmos3-Nano>
- Reference code: <https://github.com/nvidia/cosmos>

Example local layout:

```text
path\to\cosmos3-nano\weights\
├── scheduler
├── sound_tokenizer
├── text_tokenizer
├── transformer
└── vae
```

Download example:

```powershell
huggingface-cli login
huggingface-cli download nvidia/Cosmos3-Nano --local-dir path\to\cosmos3-nano\weights
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
  --model path\to\cosmos3-nano\weights `
  --tokenizer_subfolder text_tokenizer `
  --fps 24
```

Rerun this cache after changing prompt template flags or captions.

## Cache Latents

Video-only:

```powershell
python cosmos3_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae path\to\cosmos3-nano\weights `
  --vae_subfolder vae `
  --vae_dtype bfloat16
```

Video+audio:

```powershell
python cosmos3_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae path\to\cosmos3-nano\weights `
  --vae_subfolder vae `
  --vae_dtype bfloat16 `
  --cache_audio `
  --sound_tokenizer path\to\cosmos3-nano\weights `
  --sound_tokenizer_subfolder sound_tokenizer `
  --sound_sample_rate 48000 `
  --sound_channels 2 `
  --sound_latent_fps 25
```

If video-only caches already exist, rerun latent caching without `--skip_existing` to add `sound_latents_*`.

## Cache Reasoner K/V (optional)

Each decoder layer holds a frozen "reasoner" tower beside the trained generation
tower. The reasoner reads only the caption, so its K/V does not depend on the
latent, timestep, resolution or frame count. Caching it per caption lets training
skip loading those ~8B parameters, bit-identical to a normal bf16 run.

```powershell
python -m musubi_tuner.cosmos3_cache_reasoner_kv `
  --dataset_config path\to\dataset.toml `
  --dit path\to\cosmos3-nano\weights `
  --reasoner_kv_cache_dir path\to\reasoner_cache `
  --sample_prompts path\to\sample_prompts.txt `
  --fps 24
```

Then add `--reasoner_kv_cache_dir path\to\reasoner_cache` when training.

| DiT parameters | Without | With |
| --- | --- | --- |
| bf16 | 28.3 GiB | 15.3 GiB |
| `--fp8_scaled` | 15.3 GiB | 8.9 GiB |

Disk cost is ~144 KiB per text token: ~33 MiB for a 235-token caption, ~3.2 GiB
per 100 captions. Entries are content-addressed by token IDs, so cost scales with
*unique* captions, not samples.

Pass `--sample_prompts` if training uses it; sampling needs its own prompts
cached, including the CFG unconditional branch. Training checks for them at
startup.

The cache key is the tokenized caption, so `--fps` and the `--no_*_template`
flags must match between caching and training. Rerun this cache after changing
captions or prompt template flags, as with the text token cache.

Works with `--fp8_scaled`, `--blocks_to_swap` and `--gradient_checkpointing`.
Multi-GPU context parallel is not supported.

## Train

Video+audio LoRA, scaled FP8 base, gradient checkpointing, block swap, samples/checkpoints every 300 steps:

```powershell
python -m accelerate.commands.launch cosmos3_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit path\to\cosmos3-nano\weights `
  --vae path\to\cosmos3-nano\weights `
  --transformer_subfolder transformer `
  --vae_subfolder vae `
  --tokenizer_subfolder text_tokenizer `
  --sound_tokenizer path\to\cosmos3-nano\weights `
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

## Inference

Text-to-video:

```powershell
python cosmos3_generate_video.py `
  --dit path\to\cosmos3-nano\weights `
  --vae path\to\cosmos3-nano\weights `
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
  --dit path\to\cosmos3-nano\weights `
  --vae path\to\cosmos3-nano\weights `
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

Add `--audio --sound_tokenizer path\to\cosmos3-nano\weights --sound_tokenizer_subfolder sound_tokenizer` to decode AVAE audio and write a WAV next to the video.
