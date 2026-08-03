# MiniMax H3

MiniMax H3 support uses the same architecture-specific script layout as other Musubi models. Model checkpoints are treated as data;
they never select or execute Python code.

The implementation models a 50-block packed transformer that jointly processes video and stereo audio. Its media contract is:

- 24 fps video with frame counts on the `17k+5` grid;
- 24-channel video latents, spatial compression 16, and a `(1, 2, 2)` transformer patch;
- 32 kHz stereo audio with 32-channel latents at 40 latent frames per second;
- synchronized video and audio flow schedules with shifts 12.0 and 3.0;
- 5120-wide conditioning states;
- optional image, video, and audio reference context.

These constants follow the public [ComfyUI integration](https://github.com/Comfy-Org/ComfyUI/pull/15224) and
[Diffusers integration](https://github.com/huggingface/diffusers/pull/14355). The backend must validate them against the selected
checkpoint configuration and reject incompatible layouts.

## Checkpoint inspection

The inspector reads safetensors headers without allocating tensor storage:

```shell
python minimax_h3_generate_video.py \
  --model /path/to/MiniMax-H3 \
  --prompt "A product reveal in a studio" \
  --output output.mp4 \
  --inspect
```

It reports checkpoint components, shards, tensor counts, parameter counts, dtypes, key prefixes, and configuration files.

## Dataset and cache layout

H3 reuses Musubi's standard video dataset and control fields. A target video's embedded soundtrack is the synchronized audio target.
When a target video has no audio stream, latent caching encodes duration-matched silence and marks its entire audio loss mask invalid,
so both `fl2va` and `ref2va` can train on video-only examples. A present but corrupt audio stream remains an error. Reference image,
video, or audio assets use `control_directory`, `control_path`, or numbered `control_path_N` fields; no H3-specific dataset schema is
required.

The released processor uses a 768-pixel short edge with a 1344x768 area cap. Other 32-pixel-aligned dimensions are structurally
valid, but should be treated as experimental training canvases.

```toml
[general]
resolution = [1344, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
video_directory = "/path/to/videos"
cache_directory = "/path/to/cache"
target_frames = [124, 175, 243, 294, 362]
frame_extraction = "uniform"
```

Cache commands preserve Musubi's normal traversal, bucketing, skip/keep behavior, and cache cleanup:

```shell
python minimax_h3_cache_latents.py \\
  --dataset_config dataset.toml \\
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \\
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors
python minimax_h3_cache_text_encoder_outputs.py \\
  --dataset_config dataset.toml \\
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \\
  --tokenizer /path/to/MiniMax-H3/FL2VA/text_encoder \\
  --task t2va
```

The native loader preserves the released component precision: the Comfy video encoder is FP16 and the audio encoder is FP32.
Normalized cache tensors default to float32, and target video/audio caches use the posterior mean for deterministic reuse. Cache
filenames use Musubi's `mmh3` architecture short name. The latent cache contains normalized video latents, normalized stereo audio
latents as `latents_audio_2x32xT_*` in stereo-major `[2, 32, T]` layout, and explicit loss masks. The conditioning cache is
crop-specific and uses `varlen_mmh3_*` tensors. `--task t2va` stores the raw-caption layer-50 output; `--task fl2va` adds the selected
crop's first and last images, including their vision rows and modality tags. Empty-text conditioning is stored only when
`--cache_guidance_empty` is requested.

## Backend boundary

The H3 implementation is native PyTorch adapted from the Apache-licensed public Diffusers integration; model checkpoints are never
allowed to select or execute Python code. `src/musubi_tuner/minimax_h3/integration.py` is the component adapter. Its factories keep
latent caching, conditioning caching, generation, and training on separate loading paths.

| Script | Integration factory | Components loaded |
| --- | --- | --- |
| `minimax_h3_cache_latents.py` | `create_latent_encoder` | Video VAE and audio VAE |
| `minimax_h3_cache_text_encoder_outputs.py` | `create_conditioning_encoder` | Understanding encoder and its processor |
| `minimax_h3_generate_video.py` | `create_generator` | Only the transformer variant and decoding components required by the request |
| `minimax_h3_train_network.py` | `create_training_backend` | Only the selected `fl2va` or `ref2va` transformer; latents and conditioning come from caches |

The adapter must use strict checkpoint loading and explain every supported key conversion. H3-specific quantization, block
swapping, and compilation remain disabled until the released checkpoint keys and module boundaries can be validated with real
forward and backward passes.

## LoRA training

The H3 trainer follows Musubi's `NetworkTrainer` and LoRA module contracts. The default `fl2va` mode uses the released base
transformer, which handles text-only, first-frame, last-frame, and first-and-last-frame conditioning. It jointly noises cached video
and audio, maps the video sigma onto the synchronized audio schedule, and applies a masked joint velocity loss. Attention and
feed-forward projections are adapter targets; norms and timestep/modality calibration remain frozen.

The default `token` loss mode is an element-weighted reduction over valid latent values; `modality` gives the video and audio means
equal weight before applying their explicit loss weights. LoHa/LoKr architecture detection remains unavailable until the released
backend provides its exact class and state-dict mapping.

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /path/to/MiniMax-H3 \
  --dataset_config dataset.toml \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --network_dim 16 --network_alpha 16 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --output_dir output --output_name h3_style
```

`--h3_training_mode ref2va` selects a separate reference-conditioned transformer when the backend provides one. Reference training
requires distinct context and target media; a target must not be reused as its own reference.

The official model card identifies the released weights as CFG-distilled. H3 inference uses one model evaluation per sampling step,
without a negative-prompt branch. Normal LoRA training therefore uses one conditional
forward and the model's joint flow target. If an authoritative distillation scale is known, `--h3_guidance_distillation_scale` can
enable an experimental two-pass guidance-consistent objective using cached empty-text conditioning. This does not reconstruct an
unconditional model or add negative-prompt inference.
