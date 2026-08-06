# MiniMax H3

MiniMax H3 support uses the same architecture-specific script layout as other Musubi models. Model checkpoints are treated as data;
they never select or execute Python code.

> [!WARNING]
> This branch is an experimental proof of concept associated with [issue #106](https://github.com/AkaneTendo25/musubi-tuner/issues/106).
> A separate implementation is being developed for upstream Musubi in [kohya-ss/musubi-tuner PR #1018](https://github.com/kohya-ss/musubi-tuner/pull/1018).
> Users who need future mainline support should follow that PR.
>
> **This document is rewritten continuously and describes a moving target.** The branch is under active development: checkpoint
> formats, command-line flags, defaults, dataset fields, and LoRA compatibility all change without deprecation periods while H3
> training quality is being validated. Anything here may be outdated, superseded, or removed. Pin a commit if you need stability,
> and re-read this page after every update rather than relying on remembered behaviour.

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

For FL2VA text-to-video or image-to-video training, the native loaders use the transformer, both VAEs, and one of the two
Comfy-Org conditioner checkpoints below:

- [FL2VA BF16 transformer](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors),
  `diffusion_models/minimax_h3_fl2va_bf16.safetensors`;
- [FL2VA pruned INT8 ConvRot transformer](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors),
  `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`;
- [Qwen3-VL BF16 text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors),
  `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`;
- [Qwen3-VL NVFP4/AWQ text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors),
  `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`;
- [FP16 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors),
  `vae/minimax_h3_video_vae_fp16.safetensors`;
- [FP32 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors),
  `vae/minimax_h3_audio_vae_fp32.safetensors`.

Ref2VA training uses the same conditioner and VAEs but selects the separate released
[Ref2VA BF16 transformer](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors),
`diffusion_models/minimax_h3_ref2va_bf16.safetensors`. The loader validates the requested variant and does not alias a base FL2VA
checkpoint to Ref2VA.

The official H3 tokenizer, Qwen3-VL configuration, and image/video processor metadata are bundled as package assets. The commands
therefore do not download the 14 official text-model shards and do not require `--tokenizer`. An explicit `--tokenizer` path remains
available when testing different processor metadata.

With `huggingface-hub` installed, download only the T2VA files instead of the complete model repositories:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir /models/MiniMax-H3

```

For Ref2VA, download its transformer in addition to the common files:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --local-dir /models/MiniMax-H3
```

To use the directly quantized conditioner, download it instead of the BF16 text encoder:

```shell
hf download Comfy-Org/MiniMax-H3 \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --local-dir /models/MiniMax-H3
```

To train with the pruned INT8 ConvRot transformer, download it instead of the BF16 transformer:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --local-dir /models/MiniMax-H3
```

The resulting layout used by the commands below is:

```text
/models/MiniMax-H3/
├── diffusion_models/
│   ├── minimax_h3_fl2va_bf16.safetensors
│   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors  # optional compressed alternative
│   └── minimax_h3_ref2va_bf16.safetensors       # Ref2VA only
├── text_encoders/
│   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors  # optional quantized alternative
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
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

The complete workflow is: install Musubi, download the required component/metadata groups, create the dataset configuration,
cache latents, cache Qwen conditioning, launch training, inspect TensorBoard, and evaluate checkpoints through training-time or
standalone sampling. Latent and conditioning caches must both be created before training; the trainer does not load either VAE or
Qwen merely to reconstruct missing caches. Image-only training does not require the audio VAE.

H3 reuses Musubi's standard video dataset and control fields. A target video's embedded soundtrack is the synchronized audio target.
When a target video has no audio stream, latent caching encodes duration-matched silence and marks its entire audio loss mask invalid,
so T2VA can train on video-only examples. A present but corrupt audio stream remains an error. Reference image,
video, or audio assets use `control_directory`, `control_path`, or numbered `control_path_N` fields; no H3-specific dataset schema is
required.

Still-image training likewise reuses Musubi's standard `image_directory` or `image_jsonl_file` fields. Each target is encoded
directly as one causal video-VAE frame and produces no target-audio cache. Image and video datasets may coexist in one training
configuration; joint AV/video soundtracks, audio-only targets, and audio references require `--audio_vae` while caching latents.

True modality-only training is explicit and does not use masked dummy targets. Set `h3_target_mode = "video"` on a normal
video dataset to omit audio decoding, audio caching, and audio transformer rows. For audio-only training, set
`h3_target_mode = "audio"` and provide either `audio_directory` (same-stem `.txt` captions) or `audio_jsonl_file` (records with
`audio_path` and `caption`). Audio-only datasets reuse `target_frames` to define duration on H3's 24-fps temporal grid; currently
exactly one valid `17k+5` value is required. Their `resolution` defines only the latent canvas used for H3's audio rotary positions.
Use `--task t2va` for both modality-only modes. Video-only latent caching omits `--audio_vae`; audio-only caching omits `--vae`.

### Cross-modal training

`h3_target_mode` removes a modality from the sequence entirely. `--h3_observed_modality` instead keeps both modalities packed and
trains one of them while the other is read as conditioning:

| flag | generated | observed | task |
| --- | --- | --- | --- |
| `--h3_observed_modality video` | audio | video | video-to-audio, Foley, dubbing |
| `--h3_observed_modality audio` | video | audio | audio-driven video |
| `--h3_observed_modality random` | redrawn per step | redrawn per step | one adapter covering joint, A2V and V2A |

```shell
accelerate launch minimax_h3_train_network.py ... --h3_observed_modality video
```

The observed modality is pinned to the noise level the released transformer already uses for conditioning of that kind: video
conditioning carries a trace of noise at timestep 0.999, and audio conditioning passes through untouched at timestep 1.0. Because
H3 forms its inputs as `(1 - sigma) * x0 + sigma * noise`, both are reached by fixing that modality's sigma rather than by any
separate code path, so an observed modality is presented exactly as reference media of the same modality already is.

`random` redraws the task each step across joint generation, video-observed and audio-observed in equal proportion, so a single
adapter keeps all three instead of specialising on whichever task was fixed for the run. Validation reports the joint objective
rather than a different draw each time, so its numbers stay comparable across runs.

The observed modality carries no training signal: its loss weight is forced to zero, overriding `--h3_video_loss_weight` or
`--h3_audio_loss_weight`. Sample weighting keys on the generated modality's sigma, since the observed one no longer follows the
sampled schedule. Datasets must cache both modalities (`h3_target_mode = "av"`, the default); batches carrying a single modality
are rejected rather than silently trained as ordinary joint targets.

The released processor uses a 768-pixel short edge with a 1344x768 area cap. Other 32-pixel-aligned dimensions are structurally
valid, but are outside the released processor's native canvas distribution.

Complete dataset examples are provided for every implemented training task:

| Training task | Dataset example | Transformer | Text-cache task | Training mode |
| --- | --- | --- | --- | --- |
| Text-conditioned image | [`image.toml`](../examples/minimax_h3/image.toml) | FL2VA | `t2va` | `fl2va` (default) |
| Text-conditioned T2VA | [`t2va.toml`](../examples/minimax_h3/t2va.toml) | FL2VA | `t2va` | `fl2va` (default) |
| Video-only T2V | [`video_only.toml`](../examples/minimax_h3/video_only.toml) | FL2VA | `t2va` | `fl2va` (default) |
| Audio-only T2A | [`audio_only.toml`](../examples/minimax_h3/audio_only.toml) | FL2VA | `t2va` | `fl2va` (default) |
| First-frame I2V | [`i2va.toml`](../examples/minimax_h3/i2va.toml) | FL2VA | `i2va` | `fl2va` (default) |
| First+last-frame video | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml) | FL2VA | `fl2va` | `fl2va` (default) |
| Arbitrary references | [`ref2va.toml`](../examples/minimax_h3/ref2va.toml) | Ref2VA | `ref2va` | `ref2va` |
| Mixed zero-or-more references (experimental) | [`ref2va_omni.toml`](../examples/minimax_h3/ref2va_omni.toml) | Ref2VA | `ref2va_omni` | `ref2va_omni` |

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
  --task t2va \\
  --device cuda
```

For an image-only dataset, use the same commands but omit `--audio_vae` from latent caching. Use `--task t2va` for its text cache:

```shell
python minimax_h3_cache_latents.py \
  --dataset_config examples/minimax_h3/image.toml \
  --vae /models/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \
  --device cuda
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config examples/minimax_h3/image.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task t2va --device cuda
```

For Ref2VA, the latent-cache command is unchanged and automatically encodes attached references. Change the text-cache command to
`--task ref2va`; it builds the ordered reference presentation and caches its Qwen3-VL vision/text rows.

Experimental `ref2va_omni` training uses the same released Ref2VA checkpoint and packing, but permits each item to contain zero,
one, or multiple ordered references. Cache its conditioning with `--task ref2va_omni`, then train with
`--h3_training_mode ref2va_omni`. A zero-reference item is packed as `[text | target audio | target video]`; non-empty items keep
the released Ref2VA layout. This mode does not reinterpret one or two arbitrary references as FL2VA's native first/last keyframe
anchors, and it does not combine the FL2VA and Ref2VA transformers. It is an experimental way to teach one Ref2VA adapter across
mixed conditioning presentations; output quality and semantic equivalence to native FL2VA still require sampling evaluation.

For ordinary first-frame I2V, use the same FL2VA transformer and latent-cache command, but cache text conditioning with
`--task i2va`. Use `--task fl2va` for first+last-frame conditioning. Both modes take their keyframes from the target video itself;
they do not use `control_directory` or the Ref2VA reference fields.

```shell
# First-frame I2V
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task i2va --device cuda

# Experimental first+last-frame FL2VA
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
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

The released NVFP4/AWQ conditioner is loaded directly with:

```shell
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq \
  --task t2va --device cuda
```

This path keeps the 350 language-model Linear layers in their packed NVFP4/AWQ representation and the token embedding in row-wise
INT8. Each Linear weight is dequantized only for its BF16 operation, so the full dense BF16 text encoder is never materialized.
Vision modules and cached layer-50 hidden states remain BF16. The mode is explicit and accepts only the matching checkpoint format.

The pruned transformer replaces the full timestep MLP and 2,688-wide AdaLN inputs with a sampled timestep table and narrow AdaLN
projections. Its eligible attention and feed-forward Linear weights remain frozen in the checkpoint's row-wise INT8 ConvRot form.
Use it for LoRA training with:

```shell
python minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --int8_convrot_base --sdpa --mixed_precision bf16 \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16
```

`--int8_convrot_base` is explicit and cannot be combined with `--fp8_base`. The loader validates the AdaLN table, every model key,
the complete marker/scale/INT8-weight sets, and each mixed-precision tensor before loading. Omitting the flag continues to resolve and
load the full BF16 checkpoint through the unchanged BF16/FP8 path.

For additional VRAM savings during training, pass `--gradient_checkpointing --gradient_checkpointing_cpu_offload`. H3 offloads the
large unified video/audio activation between transformer blocks and moves it back to each block's device for recomputation. CPU
offloading is slower than gradient checkpointing alone and should be used when sequence length would otherwise exceed available VRAM.

With `--sdpa`, `--h3_attn_auto_dispatch` prioritizes cuDNN SDPA for large maskless CUDA BF16/FP16 attention workloads. Short
sequences, padded or custom attention masks, FP32, and CPU execution retain ordinary PyTorch SDPA. The option is disabled by default
because backend selection can change numerical rounding; benchmark throughput and training quality on the intended shape.

The native loader preserves the released component precision: the Comfy video encoder is FP16 and the audio encoder is FP32.
Normalized cache tensors default to float32, and target video/audio caches use the posterior mean for deterministic reuse. Cache
filenames use Musubi's `mmh3` architecture short name. The latent cache contains normalized video latents, normalized stereo audio
latents as `latents_audio_2x32xT_*` in stereo-major `[2, 32, T]` layout, and explicit loss masks. The conditioning cache is
crop-specific and uses `varlen_mmh3_*` tensors. `--task t2va` stores the raw-caption layer-50 output; `--task i2va` adds the selected
crop's first image, and `--task fl2va` adds its first and last images, including their vision rows and modality tags. The latent
cache always stores both target keyframe rows so it can be paired with either conditioning task. `--task ref2va` stores the released ordered
`<Picture i>` / `<Audio j>` / `<Video k>` presentation, while its VAE cache stores the corresponding clean reference rows and
geometry. `--task ref2va_omni` uses the same presentation when references exist and stores text-only conditioning when they do not.
Empty-text conditioning is stored only when `--cache_guidance_empty` is requested.

## Backend boundary

The H3 implementation is native PyTorch adapted from the Apache-licensed public Diffusers integration; model checkpoints are never
allowed to select or execute Python code. `src/musubi_tuner/minimax_h3/integration.py` is the component adapter. Its factories keep
latent caching, conditioning caching, generation, and training on separate loading paths.

| Script | Integration factory | Components loaded |
| --- | --- | --- |
| `minimax_h3_cache_latents.py` | `create_latent_encoder` | Video VAE; audio VAE only for video targets or audio references |
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
jointly noise the target video and audio from one shared schedule coordinate and apply the same masked joint velocity loss.
Attention and feed-forward projections are adapter targets; norms and timestep/modality calibration stay frozen.

An image batch packs `[text | target image latent]`, has no audio rows, and applies loss only to its one target video-latent frame.
It requires the FL2VA transformer and a `--task t2va` conditioning cache. Target stills use the deterministic posterior
mean from a direct causal `T=1` VAE encode; they are not repeated into a synthetic 17-frame video clip.

The first-frame `i2va` path uses the released keyframe representation. The `fl2va` first+last variant uses the same representation
and public [ComfyUI inference contract](https://github.com/Comfy-Org/ComfyUI/blob/9a9fdb10ed144ce760d9682cb247526ea23cc525/comfy_extras/nodes_minimax_h3.py).

Ref2VA selects the dedicated released transformer with `--h3_training_mode ref2va`. References are packed in dataset order as
`[text | ordered reference blocks | target audio | target video]`. Image and video anchors use the released sampled-posterior VAE
recipe and fixed `t=0.999` noise augmentation; reference audio stays clean at `t=1`. Only generated target rows contribute to the
loss.

`--h3_training_mode ref2va_omni` is an opt-in extension of that path. It accepts matching `--task ref2va_omni` caches with
zero-or-more references while strict `ref2va` continues to require at least one reference. The standard `fl2va`, `i2va`, and
`ref2va` contracts are unchanged.

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

### CREPA

CREPA is an optional temporal representation-alignment loss for video training. Backbone mode aligns projected features from an
earlier transformer block with detached features from a later block. DINO mode aligns them with frozen per-frame DINOv2 patch
features. Both compare the same and neighboring latent frames. Only generated video rows participate: text, audio, keyframes,
and Ref2VA reference rows are excluded.

Enable the default configuration with `--crepa`, or override its fields through the same option:

```shell
accelerate launch minimax_h3_train_network.py ... \
  --crepa mode=backbone student_block=16 teacher_block=33 weight=0.05 tau=1 neighbors=2
```

All values following `--crepa` use `key=value` syntax:

| Field | Default | Meaning |
| --- | ---: | --- |
| `mode` | `backbone` | `backbone` uses a detached later H3 block as teacher. `dino` uses pre-cached DINOv2 frame features. |
| `student_block` | `16` | Zero-based transformer block whose generated-video features remain attached to the training graph and are passed through the CREPA projector. It must be non-negative and earlier than `teacher_block`. |
| `teacher_block` | `33` | Zero-based later transformer block providing the detached alignment target. It must be smaller than the transformer's 50-block count. |
| `dino_model` | `dinov2_vitb14` | DINOv2 teacher variant for `mode=dino`: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, or `dinov2_vitg14`. |
| `weight` | `0.05` | Multiplier applied to the auxiliary negative cosine-similarity objective. It must be finite and greater than zero. Larger values give representation alignment more influence relative to the normal flow loss. |
| `tau` | `1.0` | Positive temporal-distance temperature. A comparison at frame offset `d` receives weight `exp(-abs(d) / tau)`; larger values retain more influence from distant neighbors. |
| `neighbors` | `2` | Number of preceding and following latent frames compared with each student frame, in addition to its same-frame teacher target. It must be non-negative and is automatically limited by the available latent-frame count. |
| `schedule` | `constant` | Auxiliary-weight schedule: `constant`, `linear`, or `cosine`. |
| `warmup_steps` | `0` | Number of optimizer steps used to ramp the CREPA weight from zero. |
| `max_steps` | `0` | End step for linear/cosine decay. It is required for those schedules. |
| `normalize` | `true` | L2-normalize student and teacher representations before alignment. `false` uses their raw dot product. |
| `cutoff_step` | `0` | Disable CREPA at this step; zero leaves it enabled. |
| `similarity_threshold` | unset | Disable CREPA when the EMA alignment reaches this value in `[0, 1]`. |
| `similarity_ema_decay` | `0.99` | EMA decay used by the similarity threshold. |
| `threshold_mode` | `permanent` | `permanent` keeps CREPA disabled after crossing the threshold; `recoverable` reevaluates it every step. |

Bare `--crepa` enables exactly these defaults. Unknown fields, duplicate fields, malformed values, invalid block order, and
non-positive `weight` or `tau` are rejected before loading the transformer.

The option is absent by default and then installs no hooks, projection parameters, or auxiliary loss. It applies to video clips
with more than one latent frame and is skipped for image-only batches, audio-only batches, and batches where video is the observed
conditioning modality. CREPA's projection head is training state rather than part of the inference LoRA; Accelerate state saves
store it separately as `h3_crepa.safetensors` so interrupted runs can resume exactly.

For a DINOv2 teacher, create the companion caches after the normal latent cache:

```shell
python minimax_h3_cache_dino_features.py \
  --dataset_config path/to/dataset.toml \
  --dino_model dinov2_vitb14 \
  --device cuda \
  --atomic_cache_writes \
  --skip_existing

accelerate launch minimax_h3_train_network.py ... \
  --crepa mode=dino dino_model=dinov2_vitb14 weight=0.05
```

The cache command downloads DINOv2 through Torch Hub by default. Use `--dino_repo` for a local DINOv2 checkout and
`--torch_hub_dir` to select a persistent model-cache directory. It stores frozen per-frame patch features next to each H3 latent
cache; the DINO model is not loaded during training. `validation_datasets` entries are cached in the same invocation. Atomic
writes prevent an interrupted extraction from leaving a partial cache at its final name. The selected `dino_model` must match
the cache.

### Flow schedules

Each step draws one unshifted schedule coordinate and derives both modality sigmas from it independently, so video and audio
always sit at the same underlying schedule position:

```
video_sigma = shift(u, --h3_shift_video)
audio_sigma = shift(u, --h3_shift_audio)
```

`--h3_shift_video` defaults to `12.0` and `--h3_shift_audio` to `3.0`, matching the released H3 schedules. Both accept any value
in `[0.01, 100.0]`, and changing one never desynchronizes the other.

Because H3 applies its own shifts, the common `--discrete_flow_shift` must stay at its default of `1.0`; passing anything else is
rejected.

`--timestep_sampling` selects the shape of the unshifted coordinate and defaults to `uniform`. Use `uniform`, `sigmoid`, or
`logsnr`: these leave the coordinate unshifted, so the per-modality shifts above are the only ones applied. `sigmoid` concentrates
it near the middle, which combined with a large video shift leaves very few steps at low sigma. The dynamic-shift modes
(`flux_shift`, `qwen_shift`, `krea2_shift`, `ideogram4_shift`, `qinglong_*`) derive a shift of their own from the latent shape and
apply it before H3's, giving a doubly-shifted schedule; they also read only the two trailing latent dimensions, so they ignore
H3's temporal extent entirely and are not meaningful here.

### Validation loss

H3 can evaluate a separate cached dataset at fixed points on the flow schedule while training:

```shell
accelerate launch minimax_h3_train_network.py ... \
  --validation_dataset_config path/to/validation.toml \
  --validate_at_start \
  --validate_every_n_steps 100 \
  --validation_timestep_bins 4 \
  --validation_min_timestep 100 \
  --validation_max_timestep 900 \
  --max_validation_items 8
```

The validation TOML uses the same dataset format, target modes, and prebuilt latent/text caches as training and must use
`batch_size = 1`. Validation is deterministic: each dataset index and timestep bin receives stable video noise, audio noise,
and model-forward RNG streams derived from `--validation_seed` (or the training seed when omitted). Constructing or running
validation restores the Python, NumPy, CPU Torch, and CUDA Torch RNG states, so enabling it does not change subsequent training
randomness.

Bins are equal-width midpoints between `--validation_min_timestep` and `--validation_max_timestep` in the shared *unshifted*
base-sigma coordinate. Video and audio then receive their exact configured H3 shifts independently. Image validation uses the
same explicit `--h3_image_flow_shift` or resolution-aware image shift as image training. Distributed workers evaluate disjoint
dataset indices and sum exact squared-error numerators and valid-element counts before logging.

TensorBoard receives `val/loss`, available `val/loss/video` and `val/loss/audio` metrics, and `val/loss/bin_NN` for every
non-empty bin. Fully masked modalities are omitted rather than reported as NaN. `val/loss` follows the primary training flow
objective, including loss balancing, modality weights, timestep weighting, observed-modality masking, and optional
guidance-consistent/contrastive prediction. It intentionally excludes CREPA and base-preservation losses: those are auxiliary
training regularizers, not held-out reconstruction quality.

### AdaLN reduction

Each block's AdaLN projection reads only the timestep. It never sees tokens or content, so its
2688-wide input traces a smooth one-dimensional curve as the timestep varies, and the projection can
be stored against a small basis of that curve rather than the full width. `--h3_adaln_rank` performs
that reduction while the checkpoint streams in, so no separate file is produced or required:

```shell
accelerate launch minimax_h3_train_network.py ... --h3_convrot_int8 --h3_adaln_rank 16
```

These projections are the largest single group of parameters in the transformer -- roughly 13.0B of
33.1B. At rank 16 they become about 77M, which changes the resident base as follows:

| base precision | default | with `--h3_adaln_rank 16` |
|---|---|---|
| BF16 | 66.2 GB | 40.5 GB |
| one byte per weight (`--h3_convrot_int8` or `--fp8_base`) | 33.1 GB | 20.4 GB |

ConvRot INT8 and scaled FP8 occupy the same space; see [Choosing a quantization](#choosing-a-quantization) for how they differ.

The reduced projections are stored in float32 and excluded from quantization: at that size
neither quantizing nor narrowing them saves anything measurable, while the stored precision is what
bounds the reduction. Storing them at the checkpoint's own BF16 would put a relative floor under the
modulation orders of magnitude above the basis error, capping every rank at the same accuracy.
Because AdaLN depends only on the timestep, any error there is a systematic distortion of the
modulation curve rather than noise that averages away across tokens. The modulation is computed in
float32 and cast back to the activation dtype, so the extra precision costs one narrow matmul.

Rank 16 reproduces the modulation to a relative error far below BF16's own rounding step, so the
reduction is not the limiting approximation anywhere in the pipeline. On the AdaLN path it is in
fact more accurate than the BF16 transformer, which computes its own modulation in BF16.

The published pruned checkpoints apply the same reduction at rank 8, storing the reduced projections
in FP16 and quantizing the rest of the transformer. `--h3_adaln_rank` instead performs the reduction
on the released BF16 transformer, leaving the choice of quantization open, and works for both
`fl2va` and `ref2va`. It composes with block swapping and gradient checkpointing, and also shrinks
the swap ring buffers, which are otherwise sized by the full-width AdaLN weights.

The option is rejected on a checkpoint that is already pruned, and on INT8 ConvRot checkpoints.

> [!NOTE]
> Measured against the BF16 transformer, forwarding identical fixed-seed inputs through all 50 blocks,
> the reduction is the most faithful of the memory-saving options and the only one that improves on
> the reference rather than departing from it:
>
> | configuration | video relative L2 | audio relative L2 |
> |---|---|---|
> | `--h3_adaln_rank 16` | **4.30e-02** | **5.71e-02** |
> | pruned INT8 ConvRot checkpoint | 8.08e-02 | 1.41e-01 |
> | `--fp8_base` | 1.25e-01 | 2.39e-01 |
>
> The middle row is the published pre-quantized checkpoint, which carries its own rank-8 FP16 AdaLN. Quantizing the released
> BF16 weights at load with `--h3_convrot_int8` is measured separately in [Choosing a quantization](#choosing-a-quantization).
>
> Every option here changes the output at the percent level, because fifty residual blocks amplify any
> small weight perturbation; that is normal rather than a defect. AdaLN reduction is roughly twice as
> close to the reference as the INT8 checkpoint and three times as close as scaled FP8, while also
> being the largest saving of the three on that tensor. Combining it with a quantization is therefore preferable to that
> quantization alone, which quantizes the AdaLN projections instead of reducing them.
>
> These figures compare forward predictions. Whether an adapter trained against one base transfers to
> a differently quantized or reduced base is a separate question that applies to every row of the
> table and has not been measured here.

Image batches are the deliberate exception. They use Musubi's existing `krea2_shift` implementation: a logit-normal base density
followed by a resolution-aware `exp(mu)` shift over the spatial DiT-token count. This image sigma is already final and is therefore
not shifted by H3's video schedule a second time. Video batches in the
same run continue to use the synchronized released 12/3 AV schedule. `--h3_image_flow_shift` optionally replaces the automatic
image shift with a fixed positive value; it does not affect video batches.

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

Use `--flash_attn` for FlashAttention 2 or `--flash3` for FlashAttention 3 on Hopper GPUs. Both are optional, require their
corresponding package, and fall back to SDPA when a batch contains explicit padding.

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
> The pruned INT8 ConvRot checkpoint now passes strict loading and training-path gradient smoke tests, but it remains experimental:
> controlled LoRA-quality and output comparisons against the full checkpoint are still required before treating the two as equivalent.

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

> [!CAUTION]
> These measurements are historical baselines and may be outdated. They do not include several newer memory and throughput
> optimizations, so they should not be interpreted as the current minimum VRAM or best achievable training speed.

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

The standalone generator uses the same native transformer forward as training. It loads the text encoder, reference encoders,
transformer, video decoder, and audio decoder sequentially, then writes a synchronized H.264/AAC MP4 and a JSON sidecar
containing the prompt, geometry, schedule, LoRA names, stage timings, and CUDA memory peaks. It supports T2VA, first- and
last-keyframe FL2VA, and arbitrary-reference Ref2VA generation.

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_fl2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
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
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A slow camera move across a quiet coastal landscape." \
  --first_frame start.png --last_frame end.png \
  --duration 5 --ratio 16:9 --steps 20 --seed 42 \
  --output conditioned.mp4
```

Ref2VA uses the dedicated Ref2VA transformer checkpoint. Add one or more `--reference_image`, `--reference_video`, or
`--reference_audio` options; a reference video's embedded soundtrack is included automatically. Standalone audio references must
be paired with at least one reference image or video.

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_ref2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A small sailboat crosses a calm bay at sunrise." \
  --reference_image subject.png \
  --duration 5 --ratio 16:9 --steps 20 --seed 42 \
  --output reference_conditioned.mp4
```

Here `--steps 20` follows the released Diffusers scheduler contract: it creates 20 sigma points including terminal zero and runs
19 transformer evaluations. Add one or more `--lora_weight` options (and matching `--lora_multiplier` values when needed) to
evaluate saved adapters. The common `--fp8_base` and block-swap options are also available for constrained inference.
Pass the pruned transformer together with `--int8_convrot_base` to use the same frozen ConvRot runtime for inference.

For samples during training, add the decoder and conditioner paths plus Musubi's common sampling options to the training command:

```shell
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
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

The guidance objective has two equivalent forms:

```text
--h3_guidance_distillation_scale 3 --h3_guidance_loss_form normalized
--h3_guidance_distillation_scale 3 --h3_guidance_loss_form contrastive
```

`normalized` reconstructs the conditional-field estimate and applies the flow loss to that estimate. `contrastive` applies the flow
loss directly to the extrapolated target `u + scale × (target - u)`. Both have the same optimum and gradient direction, but the
contrastive loss and gradient are `scale²` larger (`9×` at scale `3`). This changes its strength relative to base preservation,
weight decay, gradient clipping, and other auxiliary objectives, so their hyperparameters are not directly interchangeable.

#### Cost

Each auxiliary objective adds a full forward over the whole packed sequence:

| Enabled | Forwards per step | Notes |
| --- | --- | --- |
| neither | 1 trainable | the baseline |
| `--h3_guidance_distillation_scale` | + 1 no-grad | the empty-prompt branch |
| `--h3_base_preservation_loss_weight` | + 1 no-grad | the frozen base |
| both | 3 total | one trainable, two no-grad |

The extra passes carry no gradients, so they cost compute rather than memory: with gradient checkpointing a trainable step is
roughly four forward-equivalents, and each auxiliary branch adds one, so guidance alone is about `1.25×` a plain step and both
together about `1.5×`.

This matters most for `ref2va`. References inflate the packed sequence, and every auxiliary pass re-processes them along with the
target, so the multiplier applies to the enlarged sequence rather than the target alone. The empty-prompt branch is also not
shorter than the prompt branch -- the null conditioning is padded to the same length -- so it costs the same as the branch it
calibrates. When reference conditioning is already expensive, turning both objectives on triples the most expensive part of the
step.

### Extension training

`--h3_extension_video_frames` and `--h3_extension_audio_latents` train continuation: a leading run of the target is observed as
context and the rest is generated.

```shell
accelerate launch minimax_h3_train_network.py ... --h3_extension_video_frames 8 --h3_extension_audio_latents 20
```

The observed span is taken from the target itself, so no additional caching or dataset field is needed; an ordinary `--task t2va`
cache is enough. The packed target rows are already noised, so the context is reconstructed from the flow identity
`x0 = x_t + sigma * target` rather than sliced from them. Video context then carries the released conditioning noise level at
timestep 0.999 and audio context passes through at timestep 1.0.

`--h3_extension_route` selects how the context is presented:

| Route | Presentation | Trade-off |
| --- | --- | --- |
| `condition_rows` (default) | duplicated as extra clean rows at the coordinates of the frames they observe | the released keyframe contract generalized from one anchor to a run |
| `per_row_sigma` | pinned in place inside the target block | no extra tokens, but intra-block noise levels are outside what the released weights have seen |

`condition_rows` costs sequence length proportional to the context, which matters because attention is quadratic in it.
`per_row_sigma` avoids that entirely at the cost of leaving the released distribution. Both are available so the two can be
compared on the same data.

The observed span is removed from the loss, intersecting any mask the dataset already provides. Without that the context would
dominate the objective whenever the continuation is short, and the model would be scored on frames it was given.

Both counts are in **latent** units, not pixel frames, and each must be shorter than its target. The two are independent: setting
only one trains extension for that modality while the other is generated in full from scratch.

### Choosing a quantization

Both quantizations store the frozen base at one byte per weight, so neither has a memory advantage over the other. They differ in
what that byte buys. Measured against the same BF16 reference on FL2VA, one forward at rank 16, relative L2 of the model output:

| frozen base | video | audio |
| --- | --- | --- |
| ConvRot INT8, `--h3_convrot_int8_fwd bf16` | **6.69e-02** | **9.66e-02** |
| ConvRot INT8, fused INT8 forward | 7.85e-02 | 1.25e-01 |
| scaled FP8, per-block scale | 1.17e-01 | 2.30e-01 |
| scaled FP8, per-channel scale | 1.40e-01 | 2.19e-01 |
| scaled FP8, per-tensor scale | 1.41e-01 | 2.04e-01 |

ConvRot is the closer of the two at the same byte budget, and the rotation is what earns it: spreading the outliers lets uniform
INT8 levels carry more of the distribution than E4M3 does with its three mantissa bits. **Prefer ConvRot INT8.**

The two ConvRot rows store identical weights and differ only in how the matmul is evaluated, which is explained under
[ConvRot INT8](#convrot-int8) below. The BF16 route is the more accurate of them because it leaves the activations alone;
the fused kernel quantizes those as well.

On speed, the fused INT8 forward is the expensive one: 6.21 s/it against scaled FP8's 3.94 at 608x352x124 on one H100, both with
`--h3_adaln_rank 16`. That is a property of the kernel rather than of INT8 arithmetic, which an H100 runs at the same rate as
FP8, and it is the cost the BF16 route exists to avoid.

### ConvRot INT8

`--h3_convrot_int8` quantizes the released BF16 transformer to ConvRot INT8 while it loads, instead of reading a checkpoint that
was quantized offline:

```shell
accelerate launch minimax_h3_train_network.py ... --h3_convrot_int8 --h3_adaln_rank 16
```

ConvRot applies a block-diagonal Hadamard rotation before per-channel INT8, which spreads activation outliers so the quantization
error falls. The frozen base halves in size against BF16, and the forward runs an INT8 kernel.

Quantizing at load rather than reading a pre-quantized file matters for more than convenience. The transform hooks run first, so
this path composes with `--h3_adaln_rank`: it quantizes an AdaLN that has already been reduced, whereas the published pruned
checkpoints carry the projections at full width and quantize them there. The reduced projections are excluded from quantization
entirely, as they are on the FP8 path. That pairing is also the cheapest way to run ConvRot, since the reduced AdaLN is 13B fewer
parameters to move.

It replaces the other quantizations rather than combining with them, so `--fp8_base` and `--int8_convrot_base` are rejected
alongside it.

`--h3_convrot_int8_fwd bf16` changes how the matmul is evaluated without changing a single stored weight. The rotation is
orthogonal, so undoing it on the weight and rotating the activations into it are the same arithmetic; the first route hands the
vendor BF16 GEMM an ordinary matrix instead of running the fused INT8 kernel, and drops the gradient's rotation as well.

It is the **more accurate** of the two, which is not the trade one expects from the name. The quantization error the rotation
buys is fixed when the weight is stored, so both routes carry it identically; but the fused kernel additionally quantizes the
activations per row on every call, and the BF16 route never does. That is worth 1.17x on video and 1.30x on audio, measured
above.

`--h3_convrot_int8_bwd int8` additionally computes the backward in INT8. It is **slower** than the BF16 backward it replaces on an
H100, 8.67 s/it against 6.21, so it is off by default and worth measuring before enabling. Upstream describes the INT8 backward as
a speed win, which holds on the GPUs without FP8 support it targets.

### FP8 granularity

`--h3_fp8_quantization_mode` selects how finely the weight scale accompanying `--fp8_base` is stored: `block` (the default, one
scale per 64 elements), `channel`, or `tensor`. The granularities differ less than their names suggest, and not consistently:
per-tensor is somewhat worse on video and somewhat better on audio than per-block, as the table above shows. That is what one
would expect if the error comes mostly from E4M3's three mantissa bits rather than from the scale, so treat the choice as a small
trade rather than a quality cliff. Per-block is measurably the fastest of the three and remains the default.

Scaled FP8 stores the frozen weights in E4M3 but dequantizes them to bf16 before each matmul, so it saves memory rather than
compute. Computing the matmuls in FP8 directly was tried and withdrawn: it requires the per-tensor scale, and per-tensor already
costs more wall clock against per-block than the FP8 matmul could return.

### Reference cost

Ref2VA references inflate the packed sequence in two places at once: as transformer rows, and as Qwen3-VL vision tokens inside the
text prefix. Attention is quadratic in the packed length and the feed-forward layers linear in it, so a large reference is
expensive in both terms. A multi-second video reference can contribute several times more rows than the target it conditions.

`--reference_scales` draws a reference resolution per item at caching time:

```shell
python src/musubi_tuner/minimax_h3_cache_latents.py ... --reference_scales 1.0,0.75,0.5
```

Each reference carries its own latent height and width, independent of the target, and H3's spatial rotary grid is
area-normalized, so a smaller reference occupies the same coordinate field with fewer samples rather than a different one. Packed
rows fall roughly with the square of the scale, and because the conditioner is fed the same prepared media, its vision tokens fall
with it.

Draw a range rather than fixing one value, and keep `1.0` in the list. Generation always prepares references at the released full
resolution, so an adapter trained only on smaller ones would meet a presentation it never saw; including `1.0` keeps that case
inside the training distribution while the smaller draws still lower the average cost. Scales are bounded to `[0.25, 1.0]`, below
which too few patches remain to carry identity.

Both auxiliary objectives multiply this cost, since each adds a full forward over the same reference-inflated sequence.

### Per-frame noise spread

`--h3_frame_sigma_jitter` gives each latent frame its own noise level around the step's shared schedule position:

```shell
accelerate launch minimax_h3_train_network.py ... --h3_frame_sigma_jitter 0.2
```

One sigma per step supervises one point of the schedule per step. Spreading it across frames supervises a range of the schedule
in the same forward, which is worth most when the dataset is small. The flow target `x0 - noise` does not depend on sigma, so only
the noised input and the per-row timesteps change; the loss is untouched.

Image batches are skipped, since a single frame has nothing to spread across. `0` disables the spread and reproduces the shared
schedule exactly.

### Keyframe interpolation

`--h3_keyframe_anchors` generalizes first/last keyframe conditioning to any set of frames, training interpolation between
arbitrary anchors:

```shell
accelerate launch minimax_h3_train_network.py ... --h3_keyframe_anchors first,11,last
accelerate launch minimax_h3_train_network.py ... --h3_keyframe_random_count 3
```

Each entry is `first`, `last`, or a latent frame index. `--h3_keyframe_random_count` instead draws that many distinct anchors per
step, which trains one adapter to interpolate from whatever anchors it is later given rather than from a fixed pattern; the count
is clamped to the clip length.

Anchors are presented as the clean latents of those frames, packed a second time as condition rows at their own coordinates.
`first` and `last` keep the released coordinates -- `last` is the final *pixel* frame, which is one latent window beyond the final
window's start, so it is deliberately not the same anchor as the integer `frames - 1`. Anchor frames remain in the loss, matching
the released contract where a conditioned first frame is still predicted; a handful of anchors does not dominate the objective the
way a long observed prefix would.

This is not identical to released keyframe conditioning: the content is the target's own latent window rather than a separately
encoded keyframe image, and `--task t2va` conditioning carries no vision rows. Treat it as a related but distinct interface, and
check whether the released model already interpolates from interior anchors before training an adapter to do it.

Keyframe conditioning, extension, and masked conditioning all claim the observed rows, so only one may be enabled at a time.

### Masked conditioning

`--h3_mask_mode` hides part of the target and presents the rest as clean context, training inpainting, outpainting, and temporal
infilling:

| Mode | Hides | Task |
| --- | --- | --- |
| `box` | a spatial region, constant across time | inpainting |
| `border` | everything outside an interior crop | outpainting |
| `segment` | a contiguous run of frames | temporal infilling |

```shell
accelerate launch minimax_h3_train_network.py ... --h3_mask_mode box --h3_mask_audio
```

Masks are drawn per step rather than read from the dataset, so an ordinary `--task t2va` cache is enough and the occlusion
distribution can be changed without re-caching. `--h3_mask_min_fraction` and `--h3_mask_max_fraction` bound the fraction of each
masked axis the generated region covers. `--h3_mask_audio` additionally hides a run of audio latents.

The observed region is presented at the released conditioning noise level, as reference media already is, and the loss scores only
the generated region. One mask is drawn per step and shared by every forward that step needs, so the guidance and base-preservation
branches agree with the trainable branch about what was given.

Masks are reduced to the transformer's `(1, 2, 2)` patch grid, and a patch counts as generated when any latent inside it is. That
keeps region boundaries inside the generated set, where a seam would otherwise appear, but it also coarsens the mask: at small
latent resolutions a wide fraction range can mark every patch generated and leave nothing observed.

Masked conditioning and extension both claim the observed rows, so enabling both is rejected.

### Caption dropout

`--h3_caption_dropout_rate` replaces the prompt with the cached empty conditioning for a fraction of steps, so the adapter also
trains its unconditional branch:

```shell
accelerate launch minimax_h3_train_network.py ... --cache_guidance_empty --h3_caption_dropout_rate 0.1
```

Training every step with a prompt leaves the adapter with no null-prompt behaviour of its own, so anything that contrasts a
conditional against an unconditional prediction has nothing to contrast against. The rate is a probability in `[0, 1]`; H3 trains
one item per step, so it is a single draw per step rather than a per-sample mask, and the `h3/caption_dropped` metric records which
steps dropped.

Cached empty-text conditioning is required, so `--cache_guidance_empty` must have been passed when caching. A dropped step is
already unconditional and therefore skips the guidance-consistent correction, which has no guided field to invert; combining the
two options is allowed and simply means the correction applies to the steps that kept their prompt. When
`--h3_base_preservation_loss_weight` is active, the frozen branch follows the same conditioning as the trainable one, so a dropped
step compares like with like.

### Optional base-preservation loss

Longer LoRA runs may drift away from the behavior of H3's released guidance-distilled base. The opt-in
`--h3_base_preservation_loss_weight` objective evaluates the same conditional input once with the LoRA disabled and penalizes the
trainable prediction for moving away from that frozen-base prediction. This is function-space preservation, not CFG distillation or
de-distillation, and it does not add negative-prompt inference.

```text
--h3_base_preservation_loss_weight 0.05
```

The default is `0`, so ordinary training is unchanged. Enabling it adds one no-gradient transformer forward per microbatch. VRAM
growth should be modest, but the extra forward can be expensive with extensive block swapping because the swapped blocks must be
streamed again. The loss preserves whichever base is loaded: with FP8, INT8 ConvRot, or reduced AdaLN it anchors to that transformed
base rather than to the original BF16 checkpoint.

Community experiments by [@Ada123-a](https://github.com/Ada123-a) reported weights around `0.05`–`0.10` together with guidance scale
`3`. Treat these as preliminary starting points rather than a validated recipe: preservation strength interacts with the guidance
scale, model quantization, dataset, rank, and learning rate.
