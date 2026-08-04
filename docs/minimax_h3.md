# MiniMax H3

MiniMax H3 support uses the same architecture-specific script layout as other Musubi models. Model checkpoints are treated as data;
they never select or execute Python code.

> [!WARNING]
> This branch is an experimental proof of concept associated with [issue #106](https://github.com/AkaneTendo25/musubi-tuner/issues/106).
> A separate implementation is being developed for upstream Musubi in [kohya-ss/musubi-tuner PR #1018](https://github.com/kohya-ss/musubi-tuner/pull/1018).
> Users who need future mainline support should follow that PR. Checkpoint formats, commands, and LoRA compatibility in this branch
> may continue to change while H3 training quality is being validated.

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

## Installation and model download

Follow the repository's main [installation instructions](../README.md#installation) first. The H3 scripts require Python 3.10,
3.11, or 3.12; the profiled environment uses Python 3.12 and the dependency versions pinned by `pyproject.toml`.

For FL2VA text-to-video or image-to-video training, the native loaders use these four Comfy-Org checkpoint files:

- [FL2VA BF16 transformer](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors),
  `diffusion_models/minimax_h3_fl2va_bf16.safetensors`;
- [Qwen3-VL BF16 text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors),
  `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`;
- [FP16 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors),
  `vae/minimax_h3_video_vae_fp16.safetensors`;
- [FP32 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors),
  `vae/minimax_h3_audio_vae_fp32.safetensors`.

Ref2VA training uses the same conditioner and VAEs but selects the separate released
[Ref2VA BF16 transformer](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors),
`diffusion_models/minimax_h3_ref2va_bf16.safetensors`. The loader validates the requested variant and does not alias a base FL2VA
checkpoint to Ref2VA.

The text-cache command also needs the small tokenizer and processor metadata from the official
`FL2VA/text_encoder` directory. The 14 official text-model shards are not needed when the Comfy-Org BF16 text encoder is used. Pass
that metadata directory to `--tokenizer`; Musubi does not import ComfyUI's bundled tokenizer or hard-coded processor classes.

With `huggingface-hub` installed, download only the T2VA files instead of the complete model repositories:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir /models/MiniMax-H3

hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/text_encoder/*" \
  --exclude "FL2VA/text_encoder/model-*.safetensors" "FL2VA/text_encoder/model.safetensors.index.json" \
  --local-dir /models/MiniMax-H3-metadata
```

For Ref2VA, download its transformer in addition to the common files:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --local-dir /models/MiniMax-H3
```

The resulting layout used by the commands below is:

```text
/models/MiniMax-H3/
├── diffusion_models/
│   ├── minimax_h3_fl2va_bf16.safetensors
│   └── minimax_h3_ref2va_bf16.safetensors       # Ref2VA only
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_bf16.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
/models/MiniMax-H3-metadata/
└── FL2VA/text_encoder/                           # config/tokenizer/processor files only
```

> [!NOTE]
> `FL2VA` and `Ref2VA` describe different conditioning contracts. The FL2VA transformer supports text-only generation plus
> optional first/last keyframes; ordinary first-frame I2V therefore uses the FL2VA checkpoint. Ref2VA uses the separate transformer
> for arbitrary image, video, and audio references. This branch supports FL2VA text-only and first-frame I2V training, plus the
> arbitrary-reference Ref2VA path. First+last-frame FL2VA is also implemented, but no public first+last training recipe has been
> released; that variant currently follows the public inference packing contract and should be treated as experimental.

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

The complete workflow is: install Musubi, download the five required component/metadata groups, create the dataset configuration,
cache AV latents, cache Qwen conditioning, launch training, inspect TensorBoard, and evaluate checkpoints through training-time or
standalone sampling. Latent and conditioning caches must both be created before training; the trainer does not load either VAE or
Qwen merely to reconstruct missing caches.

H3 reuses Musubi's standard video dataset and control fields. A target video's embedded soundtrack is the synchronized audio target.
When a target video has no audio stream, latent caching encodes duration-matched silence and marks its entire audio loss mask invalid,
so T2VA can train on video-only examples. A present but corrupt audio stream remains an error. Reference image,
video, or audio assets use `control_directory`, `control_path`, or numbered `control_path_N` fields; no H3-specific dataset schema is
required.

The released processor uses a 768-pixel short edge with a 1344x768 area cap. Other 32-pixel-aligned dimensions are structurally
valid, but are outside the released processor's native canvas distribution.

Complete dataset examples are provided for every implemented training task:

| Training task | Dataset example | Transformer | Text-cache task | Training mode |
| --- | --- | --- | --- | --- |
| Text-conditioned T2VA | [`t2va.toml`](../examples/minimax_h3/t2va.toml) | FL2VA | `t2va` | `fl2va` (default) |
| First-frame I2V | [`i2va.toml`](../examples/minimax_h3/i2va.toml) | FL2VA | `i2va` | `fl2va` (default) |
| First+last-frame video | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml) | FL2VA | `fl2va` | `fl2va` (default) |
| Arbitrary references | [`ref2va.toml`](../examples/minimax_h3/ref2va.toml) | Ref2VA | `ref2va` | `ref2va` |

The first three tasks use the target video alone. Ref2VA additionally resolves ordered reference media through Musubi's existing
control fields.

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

For a directory dataset, `control_directory` contains references whose basename matches each target. For multiple ordered or
mixed-modality references, use the existing JSONL fields `control_path_0`, `control_path_1`, and so on. Indices must be contiguous
from zero because reference order controls both prompt labels and the joint rotary timeline:

```json
{"video_path":"/path/to/target.mp4","caption":"A cinematic scene","control_path_0":"/path/to/subject.png","control_path_1":"/path/to/motion.mp4","control_path_2":"/path/to/sound.wav"}
```

Cache commands preserve Musubi's normal traversal, bucketing, skip/keep behavior, and cache cleanup:

```shell
python minimax_h3_cache_latents.py \\
  --dataset_config dataset.toml \\
  --vae /models/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \\
  --audio_vae /models/MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors \\
  --device cuda
python minimax_h3_cache_text_encoder_outputs.py \\
  --dataset_config dataset.toml \\
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \\
  --tokenizer /models/MiniMax-H3-metadata/FL2VA/text_encoder \\
  --task t2va \\
  --device cuda
```

For Ref2VA, the latent-cache command is unchanged and automatically encodes attached references. Change the text-cache command to
`--task ref2va`; it builds the ordered reference presentation and caches its Qwen3-VL vision/text rows.

For ordinary first-frame I2V, use the same FL2VA transformer and latent-cache command, but cache text conditioning with
`--task i2va`. Use `--task fl2va` for first+last-frame conditioning. Both modes take their keyframes from the target video itself;
they do not use `control_directory` or the Ref2VA reference fields.

```shell
# First-frame I2V
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --tokenizer /models/MiniMax-H3-metadata/FL2VA/text_encoder \
  --task i2va --device cuda

# Experimental first+last-frame FL2VA
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --tokenizer /models/MiniMax-H3-metadata/FL2VA/text_encoder \
  --task fl2va --device cuda
```

> [!IMPORTANT]
> FL2VA keyframe training needs the target's sampled-posterior VAE keyframe rows in addition to its ordinary target latents.
> Generate both H3 caches with the same checkout used for training, and recreate them after updating this development branch.
> An I2VA/FL2VA run requires keyframe rows and rejects an incomplete cache instead of silently training as T2VA.

The default loads Qwen3-VL in BF16. Add `--text_encoder_quantization int8` or
`--text_encoder_quantization nf4` to quantize its Linear weights directly from the same BF16 checkpoint while loading.
INT8 reduces conditioner weight residency, while NF4 reduces it further. Non-Linear parameters and the cached raw layer-50 hidden
states remain BF16. Both modes are explicit memory/conditioning-precision tradeoffs rather than
numerically equivalent replacements for BF16. Quantization reduces text-caching VRAM, but the complete BF16 checkpoint remains
memory-mapped in CPU address space while weights are converted; it does not remove the host RAM/address-space requirement.

The native loader preserves the released component precision: the Comfy video encoder is FP16 and the audio encoder is FP32.
Normalized cache tensors default to float32, and target video/audio caches use the posterior mean for deterministic reuse. Cache
filenames use Musubi's `mmh3` architecture short name. The latent cache contains normalized video latents, normalized stereo audio
latents as `latents_audio_2x32xT_*` in stereo-major `[2, 32, T]` layout, and explicit loss masks. The conditioning cache is
crop-specific and uses `varlen_mmh3_*` tensors. `--task t2va` stores the raw-caption layer-50 output; `--task i2va` adds the selected
crop's first image, and `--task fl2va` adds its first and last images, including their vision rows and modality tags. The latent
cache always stores both target keyframe rows so it can be paired with either conditioning task. `--task ref2va` stores the released ordered
`<Picture i>` / `<Audio j>` / `<Video k>` presentation, while its VAE cache stores the corresponding clean reference rows and
geometry. Empty-text conditioning is stored only when `--cache_guidance_empty` is requested.

## Backend boundary

The H3 implementation is native PyTorch adapted from the Apache-licensed public Diffusers integration; model checkpoints are never
allowed to select or execute Python code. `src/musubi_tuner/minimax_h3/integration.py` is the component adapter. Its factories keep
latent caching, conditioning caching, generation, and training on separate loading paths.

| Script | Integration factory | Components loaded |
| --- | --- | --- |
| `minimax_h3_cache_latents.py` | `create_latent_encoder` | Video VAE and audio VAE |
| `minimax_h3_cache_text_encoder_outputs.py` | `create_conditioning_encoder` | Understanding encoder and its processor |
| `minimax_h3_generate_video.py` | `create_generator` | Only the transformer variant and decoding components required by the request |
| `minimax_h3_train_network.py` | `create_training_backend` | The selected `fl2va` or `ref2va` transformer; target and reference latents and conditioning come from caches |

The adapter uses strict checkpoint loading and preserves the released mixed-precision layout. Training supports Musubi's common
`--fp8_base` and block-swap options. Compilation remains unsupported.

## LoRA training

The H3 trainer follows Musubi's `NetworkTrainer` and LoRA module contracts. The default `fl2va` mode selects the released base
transformer. A `--task t2va` cache packs `[text | target audio | target video]`; `--task i2va` and `--task fl2va` insert one or two
keyframe conditions between text and target audio. Keyframes use the released fixed-seed sampled-posterior VAE encode, fixed
`t=0.999` noise augmentation, and first/last target rotary anchors. Keyframe rows never contribute to the loss. All three tasks
jointly noise the target video and audio, map the video sigma onto the synchronized audio schedule, and apply the same masked joint
velocity loss. Attention and feed-forward projections are adapter targets; norms and timestep/modality calibration stay frozen.

The first-frame `i2va` path is independently corroborated by the public
[AI Toolkit implementation](https://github.com/ostris/ai-toolkit/blob/18f5810d6c3248dc7edd8f79f3b6cc8c15c2fc98/extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py).
The `fl2va` first+last variant uses the same released keyframe representation and public
[ComfyUI inference contract](https://github.com/Comfy-Org/ComfyUI/blob/9a9fdb10ed144ce760d9682cb247526ea23cc525/comfy_extras/nodes_minimax_h3.py),
but remains experimental for training until it has a controlled real-weight backward and sampling comparison.

Ref2VA selects the dedicated released transformer with `--h3_training_mode ref2va`. References are packed in dataset order as
`[text | ordered reference blocks | target audio | target video]`. Image and video anchors use the released sampled-posterior VAE
recipe and fixed `t=0.999` noise augmentation; reference audio stays clean at `t=1`. Only generated target rows contribute to the
loss.

> [!IMPORTANT]
> Ref2VA uses the same transformer parameter count as T2VA, so its fixed weight residency is unchanged. Its activation memory can
> nevertheless be substantially higher because visual references enter the packed transformer sequence twice: as Qwen vision
> conditioning rows and as VAE reference-latent rows. A 2048x2048 image contributes 4,096 Qwen vision rows plus 4,096 VAE rows,
> or 8,192 additional packed rows. In the controlled H100 runs below, adding that reference increased BF16 + H2D48 peak allocated
> memory from 15.77 GiB to 23.05 GiB, a measured **7.28 GiB** increase. Accordingly, the measured
> 832x480x124 FP8 + H2D48 T2VA configuration at 19.40 GiB is estimated at approximately **26.7 GiB** with one square image
> reference. A 16:9 image with a 2048-pixel short edge contributes approximately 14,592 rows across both representations and is
> estimated at approximately **32–33 GiB** for the same target. Video references can cost considerably more because they add both
> temporal VAE rows and Qwen frames sampled at 2 fps; audio-reference rows are comparatively small. Values for geometries not
> present in the table remain extrapolations. Leave additional allocator/workspace headroom and validate the intended reference
> geometry before a long run.

The default `modality` loss mode gives the video and audio means equal weight before applying their explicit loss weights.
`token` is an optional element-weighted reduction over all valid latent values. LoHa/LoKr architecture detection is unsupported.

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --network_dim 16 --network_alpha 16 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --output_dir output --output_name h3_style \
  --logging_dir logs --log_with tensorboard --log_grad_metrics
```

Launch TensorBoard in a second terminal while training:

```shell
tensorboard --logdir logs
```

When training remotely, bind TensorBoard only to a protected interface or access its default loopback address through an SSH
forward. Do not expose an unauthenticated TensorBoard server directly to the public internet.

For Ref2VA, use the dedicated checkpoint and add the mode switch after caching with `--task ref2va`:

```shell
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --h3_training_mode ref2va
```

> [!CAUTION]
> Passing technical smoke tests does not establish LoRA quality. A preliminary [issue #106 community report](https://github.com/AkaneTendo25/musubi-tuner/issues/106#issuecomment-5170840323)
> observed block-like artifacts in a rank-16 concept LoRA. The cause is unresolved: overfitting, guidance-distillation breakdown,
> AdaLN quantization, data/recipe choice, and implementation error have not yet been separated experimentally. FP8 in this branch
> quantizes the frozen 13B block AdaLN projections; use BF16 as the quality reference when investigating artifacts. The released
> full checkpoint is the only training checkpoint validated here; AdaLN-pruned checkpoints remain experimental and are not accepted
> as equivalent until controlled gradient and output comparisons are complete.

`--fp8_base` enables weight-only scaled FP8 for the transformer's block `Linear` weights while keeping the surrounding model in
its normal mixed-precision dtypes. This includes the large per-block AdaLN projections. For frozen-base LoRA training, H2D-only
block swapping can be enabled with the common Musubi options:

```shell
  --fp8_base \
  --blocks_to_swap 40 \
  --block_swap_h2d_only \
  --block_swap_ring_size 2 \
  --use_pinned_memory_for_block_swap
```

Pinned host memory is strongly recommended for H2D-only swapping: it permits direct asynchronous host-to-device copies and lets
the two-buffer ring overlap transfers with transformer computation. The unpinned path remains available and uses staged copies.
H2D-only swapping is valid only while the base transformer is
frozen; the offloader checks this invariant. Block granularity keeps at least two of H3's 50 transformer blocks resident.

For the lowest transformer-weight residency, the same common offloader can stream individual frozen `Linear` layers:

```shell
  --fp8_base \
  --blocks_to_swap 50 \
  --block_swap_h2d_only \
  --block_swap_granularity layer \
  --block_swap_ring_size 2 \
  --use_pinned_memory_for_block_swap
```

Layer granularity supports offloading all 50 H3 blocks. It reloads each immutable base weight during backward while LoRA weights
remain resident and trainable. This reduces the GPU ring from complete transformer blocks to individual `Linear` payloads, at the
cost of more transfers and dispatch overhead. Use the default `block` granularity when it fits; select `layer` for extreme-offload
training.

For memory-constrained CUDA training, PyTorch's expandable allocator segments can reduce reservation fragmentation. Set
`PYTORCH_ALLOC_CONF=expandable_segments:True` before launching the process. This is a PyTorch environment setting rather than an
H3-specific command-line option.

Reference media must remain distinct from the target. Musubi reuses its existing control fields for references; it never silently
reuses the target as its own reference.

## Training benchmarks and capacity estimates

The following measurements are real LoRA optimizer steps on an NVIDIA H100 80 GB. They are intended to make memory planning
reproducible, not to predict RTX throughput. The benchmark used Python 3.12.13, PyTorch 2.9.1 with CUDA 12.8, Transformers 4.57.6,
Accelerate 1.6.0, and bitsandbytes 0.50.0. Every uncapped successful row ran 20 steps, with five warm-up steps and 15 measured
steps.

Common settings were batch size one, 124 frames at 24 fps, rank/alpha 16, AdamW8bit at `1e-4`, SDPA, gradient checkpointing,
modality-balanced joint AV loss, and a small seven-video synchronized AV dataset. Ref2VA rows use one 2048x2048 image reference
per target, contributing 4,096 VAE rows and 4,096 Qwen vision rows. All rows use the released full 33.1B transformer,
not an AdaLN-pruned checkpoint. FP8 rows quantize the frozen block AdaLN projections as well as the attention and feed-forward
linears. H2D rows use pinned host memory and the default two-buffer prefetch ring. `H2Dnn` means that `nn` of the 50 transformer
blocks are swapped.

| Weights / offload | Training mode | Resolution | Median step | Median loss T / V / A | Peak allocated VRAM |
| --- | --- | --- | ---: | ---: | ---: |
| BF16 | T2VA | 672x384x124 | 5.03 s | 0.351 / 0.350 / 0.413 | 71.16 GiB |
| BF16 | T2VA | 832x480x124 | 9.13 s | 0.280 / 0.279 / 0.381 | 75.69 GiB |
| BF16 | T2VA | 896x512x124 | — | — | OOM |
| BF16 + H2D48 | T2VA | 672x384x124 | 5.06 s | 0.366 / 0.389 / 0.398 | 15.77 GiB |
| BF16 + H2D48 | T2VA | 832x480x124 | 9.18 s | 0.336 / 0.320 / 0.379 | 20.30 GiB |
| BF16 + H2D48 | T2VA | 896x512x124 | 11.12 s | 0.312 / 0.266 / 0.389 | 22.21 GiB |
| BF16 + H2D48 | T2VA | 1120x640x124 | 21.97 s | 0.328 / 0.303 / 0.368 | 30.48 GiB |
| BF16 + H2D48 | T2VA | 1344x768x124 | 39.78 s | 0.307 / 0.254 / 0.348 | 40.60 GiB |
| FP8 | T2VA | 672x384x124 | 5.35 s | 0.353 / 0.363 / 0.398 | 42.55 GiB |
| FP8 | T2VA | 832x480x124 | 9.48 s | 0.336 / 0.302 / 0.380 | 47.08 GiB |
| FP8 | T2VA | 896x512x124 | 11.44 s | 0.312 / 0.263 / 0.376 | 48.99 GiB |
| FP8 | T2VA | 1120x640x124 | 22.25 s | 0.330 / 0.303 / 0.366 | 57.27 GiB |
| FP8 + H2D36 | T2VA | 672x384x124 | 5.49 s | 0.355 / 0.367 / 0.398 | 22.15 GiB |
| FP8 + H2D40 | T2VA | 672x384x124 | 5.47 s | 0.356 / 0.368 / 0.398 | 19.73 GiB |
| FP8 + H2D40 | T2VA | 832x480x124 | 9.50 s | 0.336 / 0.309 / 0.379 | 24.27 GiB |
| FP8 + H2D48 | T2VA | 512x288x124 | 2.92 s | 0.440 / 0.510 / 0.358 | 11.37 GiB |
| FP8 + H2D48 | T2VA | 576x320x124 | 3.67 s | 0.465 / 0.595 / 0.335 | 12.55 GiB |
| FP8 + H2D48 | T2VA | 608x352x124 | 4.31 s | 0.441 / 0.521 / 0.392 | 13.50 GiB |
| FP8 + H2D48 | T2VA | 832x480x124 | 9.57 s | 0.337 / 0.302 / 0.380 | 19.40 GiB |
| BF16 + H2D48 | Ref2VA, 1 image | 672x384x124 | 12.03 s | 0.305 / 0.271 / 0.340 | 23.05 GiB |
| FP8 + H2D48 | Ref2VA, 1 image | 672x384x124 | 12.33 s | 0.307 / 0.266 / 0.346 | 22.19 GiB |

All gradient tensors were finite in every completed run. An FP8 + H2D48 832x480x124 run also completed 20 optimizer steps under
a 22.96 GiB allocator cap. It peaked at 19.44 GiB allocated and 22.54 GiB reserved; all 400 LoRA gradient tensors were finite and
all were nonzero from step three onward. Under a 15.44 GiB allocator cap, FP8 + H2D48 completed optimizer steps at
512x288x124, 576x320x124, 608x352x124, and 640x352x124. A one-buffer ring saves approximately 0.6 GiB but was 24–35% slower in
these small-resolution tests, so it is not the default.

Dataset item count does not determine peak VRAM. Peak VRAM is set by the largest batch geometry; item count and repeats determine
steps per epoch and total runtime. For a target with requested frame count `F`, first align it to H3's frame grid:

```text
k = ceil((F - 5) / 17)
aligned_frames = 17k + 5
video_latent_frames = 5k + 2
video_tokens = video_latent_frames * (height / 32) * (width / 32)
audio_tokens = 2 * round(aligned_frames / 24 * 40)
target_AV_tokens = video_tokens + audio_tokens
```

References add their packed visual and audio rows, and text adds its Qwen rows. For cached Ref2VA data, these row counts can be
read exactly from the cache instead of inferred. Memory-efficient attention and gradient checkpointing keep measured activation
memory nearly linear in packed sequence length. For the table above, let `K = target_AV_tokens / 1000` and `B` be the number of
H2D-swapped blocks. Least-squares fits give these preliminary H100 peak-allocated estimates in GiB:

```text
BF16:  VRAM_allocated ≈ 62.52 - 1.154 * B + 0.888 * K
FP8:   VRAM_allocated ≈ 33.90 - 0.575 * B + 0.888 * K
```

The BF16 swap coefficient is validated only at `B=0` and `B=48`; the FP8 coefficient is covered at `B=0, 36, 40, 48`. These fits
use the T2VA rows only and describe this implementation and benchmark environment, not a model-level guarantee. The measured
single-image Ref2VA delta agrees with adding both its Qwen and VAE rows to `K`, but other reference kinds and shapes still require
validation. Leave roughly 15–20% above predicted allocated memory for CUDA context, allocator reserve, and shape-dependent
workspace.

At 124 frames, the BF16/H2D48 H100 step-time fit across 672x384 through 1344x768 is:

```text
median_step_seconds ≈ 0.117 + 0.3195 * K + 0.01942 * K^2
```

Its mean absolute percentage error on those five measured points is below 0.2%. FP8 was approximately 4% slower in matched tests,
while pinned two-buffer H2D swapping was within measurement noise of the corresponding resident-weight runs on this H100. The
quadratic term reflects attention compute. This timing equation must not be used as an RTX 4090/5090 prediction until measurements
from those GPUs calibrate their compute and host-transfer coefficients.

Cache profiling on one 832x480x124 item measured 17.52 seconds and 17.94 GiB peak process VRAM for joint video/audio latent
caching. BF16 T2VA text caching measured 3.30 seconds after model load and 55.72 GiB peak process VRAM. The corresponding cache
files occupied 5,594,727 bytes for AV latents and 1,169,296 bytes for conditioning; conditioning cache size varies with prompt and
reference length.

## Inference and sampling during training

The standalone generator uses the same native transformer forward as training. It loads the text encoder, keyframe VAE encoder,
transformer, video
decoder, and audio decoder sequentially, then writes a synchronized H.264/AAC MP4 and a JSON sidecar containing the prompt,
geometry, schedule, LoRA names, stage timings, and CUDA memory peaks. It supports T2VA plus first- and last-keyframe FL2VA
generation. Arbitrary-reference Ref2VA generation is not yet exposed by the standalone script.

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_fl2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --tokenizer /path/to/MiniMax-H3/FL2VA/text_encoder \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A traveler walks through a sunlit mountain valley while birds sing in the distance." \
  --duration 5 --ratio 16:9 --steps 20 --seed 42 \
  --output output.mp4
```

Add `--first_frame start.png` for ordinary first-frame I2V. Add `--last_frame end.png` to condition the final frame as well:

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_fl2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --tokenizer /path/to/MiniMax-H3-metadata/FL2VA/text_encoder \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A slow camera move across a quiet coastal landscape." \
  --first_frame start.png --last_frame end.png \
  --duration 5 --ratio 16:9 --steps 20 --seed 42 \
  --output conditioned.mp4
```

Here `--steps 20` follows the released Diffusers scheduler contract: it creates 20 sigma points including terminal zero and runs
19 transformer evaluations. Add one or more `--lora_weight` options (and matching `--lora_multiplier` values when needed) to
evaluate saved adapters. The common `--fp8_base` and block-swap options are also available for constrained inference.

For samples during training, add the decoder and conditioner paths plus Musubi's common sampling options to the training command:

```shell
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --tokenizer /path/to/MiniMax-H3/FL2VA/text_encoder \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_at_first --sample_every_n_steps 50
```

An H3 prompt-file entry can use the shared Musubi syntax:

```text
A traveler walks through a sunlit mountain valley while birds sing in the distance. --w 672 --h 384 --f 124 --d 42 --s 20
```

For first-frame or first+last-frame sampling during training, add the shared `--i` and `--ei` image options:

```text
A slow camera move across a quiet coastal landscape. --i start.png --ei end.png --w 672 --h 384 --f 124 --d 42 --s 20
```

Samples are written to `OUTPUT_DIR/sample` as synchronized MP4 files with JSON metric sidecars. Sampling uses the LoRA currently
attached to the training transformer, so identical prompt and seed settings provide a direct step-by-step learning trace. The
transformer remains reusable after sampling but is evacuated to CPU before decoding, including any active block-swap ring. The video
and audio decoders are then moved to the GPU one at a time, after which the inference ring is rebuilt and normal training swap state
is restored.

The official model card identifies the released weights as CFG-distilled. H3 inference uses one model evaluation per sampling step,
without a negative-prompt branch. Normal LoRA training therefore uses one conditional
forward and the model's joint flow target. If an authoritative distillation scale is known, `--h3_guidance_distillation_scale` can
enable an optional two-pass guidance-consistent objective using cached empty-text conditioning. This does not reconstruct an
unconditional model or add negative-prompt inference.
