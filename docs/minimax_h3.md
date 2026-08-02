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
Reference image, video, or audio assets use `control_directory`, `control_path`, or numbered `control_path_N` fields; no H3-specific
dataset schema is required.

The public inference integrations default to a 1344x768 canvas, while their model path accepts other 32-pixel-aligned dimensions.
The 832x480 example below is therefore grid-valid, although the official training-resolution distribution is not yet public.

```toml
[general]
resolution = [832, 480]
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
python minimax_h3_cache_latents.py --dataset_config dataset.toml --model /path/to/MiniMax-H3
python minimax_h3_cache_text_encoder_outputs.py --dataset_config dataset.toml --model /path/to/MiniMax-H3
```

Both VAEs default to float32 during caching. The latent cache contains normalized video latents, normalized stereo audio latents in
stereo-major `[2, 32, T]` layout, and explicit loss masks. The backend converts native layouts such as `[B, 32, 2, T]` at the cache
boundary. The conditioning cache contains variable-length hidden states and token tags. Empty-text conditioning is stored only when
`--cache_guidance_empty` is requested.

## Backend boundary

Upstream model source and its license belong under `src/musubi_tuner/minimax_h3/vendor/official/`. Musubi-specific checkpoint
loading, tensor conversion, cache conversion, and training behavior remain in the surrounding H3 modules.
`src/musubi_tuner/minimax_h3/integration.py` is the single adapter between those two layers. Its component-specific factories keep
latent caching, conditioning caching, generation, and training on separate loading paths.

| Script | Integration factory | Components loaded |
| --- | --- | --- |
| `minimax_h3_cache_latents.py` | `create_latent_encoder` | Video VAE and audio VAE |
| `minimax_h3_cache_text_encoder_outputs.py` | `create_conditioning_encoder` | Understanding encoder and its processor |
| `minimax_h3_generate_video.py` | `create_generator` | Only the transformer variant and decoding components required by the request |
| `minimax_h3_train_network.py` | `create_training_backend` | Only the selected `t2va` or `ref2va` transformer; latents and conditioning come from caches |

The adapter must use strict checkpoint loading and explain every supported key conversion. H3-specific quantization, block
swapping, and compilation remain disabled until the released checkpoint keys and module boundaries can be validated with real
forward and backward passes.

## LoRA training

The H3 trainer follows Musubi's `NetworkTrainer` and LoRA module contracts. The default `t2va` mode jointly noises cached video and
audio, maps the video sigma onto the synchronized audio schedule, and applies a masked joint velocity loss. Attention and
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

H3 inference uses one guidance-distilled model evaluation per sampling step. Normal LoRA training therefore uses one conditional
forward and the model's joint flow target. If an authoritative distillation scale is known, `--h3_guidance_distillation_scale` can
enable an experimental two-pass guidance-consistent objective using cached empty-text conditioning. This does not reconstruct an
unconditional model or add negative-prompt inference.
