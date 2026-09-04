# MiniMax H3

LoRA training and inference for MiniMax H3, a 50-block transformer that generates video and stereo audio jointly.

Media contract: 24 fps video on the `17k+5` frame grid, 24-channel latents at spatial compression 16 with a `(1, 2, 2)` patch;
32 kHz stereo audio, 32-channel latents at 40 latent frames per second; video and audio flow shifts 12.0 and 3.0.

Two released transformers, with different conditioning contracts:

| Checkpoint | Covers |
| --- | --- |
| **FL2VA** | text-to-video, first-frame I2V, first+last keyframes, last-frame L2V, video-only, audio-only, still images |
| **Ref2VA** | arbitrary image, video, and audio references |

## Contents

- [Quick start](#quick-start)
- [Workflow chooser](#workflow-chooser)
- [Model download](#model-download)
- [Train FL2VA](#train-fl2va)
  - [Choose an FL2VA task](#choose-an-fl2va-task)
  - [Prepare an FL2VA dataset](#prepare-an-fl2va-dataset)
  - [Cache FL2VA latents](#cache-fl2va-latents)
  - [Cache FL2VA text embeddings](#cache-fl2va-text-embeddings)
  - [Start FL2VA training](#start-fl2va-training)
- [Train Ref2VA](#train-ref2va)
  - [Prepare a Ref2VA dataset](#prepare-a-ref2va-dataset)
  - [Cache Ref2VA latents](#cache-ref2va-latents)
  - [Cache Ref2VA text embeddings](#cache-ref2va-text-embeddings)
  - [Start Ref2VA training](#start-ref2va-training)
- [Dataset](#dataset)
  - [Qwen control visuals](#qwen-control-visuals)
- [Pre-caching](#pre-caching)
- [Training](#training)
  - [Train a learned context](#train-a-learned-context)
  - [Use a learned context](#use-a-learned-context)
  - [Train a slider LoRA](#train-a-slider-lora)
  - [Start LoRA training](#start-lora-training)
  - [Optimizers](#optimizers)
  - [Training a guidance-distilled model](#training-a-guidance-distilled-model)
  - [Rollout supervision](#rollout-supervision)
  - [Current recommended settings](#current-recommended-settings)
  - [Full-parameter BF16 training](#full-parameter-bf16-training)
  - [Saving and resuming](#saving-and-resuming)
  - [Key options](#key-options)
  - [Memory and speed](#memory-and-speed)
  - [Training modes](#training-modes)
  - [Auxiliary objectives](#auxiliary-objectives)
  - [Differential Output Preservation](#differential-output-preservation)
  - [Validation](#validation)
  - [Sampling during training](#sampling-during-training)
- [Inference](#inference)
- [Training dashboard](#training-dashboard)

## Quick start

> [!TIP]
> Upstream [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) also supports MiniMax H3. Try the upstream implementation first unless you specifically need this fork's H3 extensions or dashboard.

1. Follow the upstream Musubi Tuner [installation instructions](https://github.com/kohya-ss/musubi-tuner#installation), including its Python and PyTorch requirements.
2. Download the checkpoints in [Model download](#model-download).
3. Configure Accelerate as described in the upstream [usage guide](https://github.com/kohya-ss/musubi-tuner#configuration-of-accelerate).
4. Follow one complete workflow: [Train FL2VA](#train-fl2va) or [Train Ref2VA](#train-ref2va). Each starts with a TOML dataset and includes both cache commands and the matching training command.
5. Use [Dataset](#dataset), [Key options](#key-options), and [Memory and speed](#memory-and-speed) as reference material after the basic workflow works. The optional [training dashboard](#training-dashboard) covers the common target layouts and explicit image/video/audio source directories; use TOML for the complete modality matrix.

This page documents only the MiniMax H3 files, contracts, and commands that differ from upstream Musubi Tuner.

## Workflow chooser

Each training objective has a fixed dataset, conditioning-cache, and transformer contract. `--task` is passed to
`minimax_h3_cache_text_encoder_outputs.py`; the final column lists any additional required cache or trainer arguments.
“None” means that no additional arguments are required.

| Training objective | Example and dataset contract | Transformer | Cache `--task` | Additional required arguments |
| --- | --- | --- | --- | --- |
| Text-to-image | [`image_fl2va.toml`](../examples/minimax_h3/image_fl2va.toml): `target_image_directory` or `image_jsonl_file`, with `target_modalities = ["image"]` | FL2VA | `t2va` | None |
| First-image-conditioned image editing | [`image_fl2va_first.toml`](../examples/minimax_h3/image_fl2va_first.toml): one basename-matched control per target | FL2VA | `fl2va` | Both cache commands: `--h3_image_mode first` |
| First+last-conditioned image editing | [`image_fl2va_first_last.toml`](../examples/minimax_h3/image_fl2va_first_last.toml): two ordered controls per target | FL2VA | `fl2va` | Both cache commands: `--h3_image_mode first_last` |
| Text-to-video+audio | [`t2va.toml`](../examples/minimax_h3/t2va.toml): `target_video_directory` or `video_jsonl_file`, with `target_modalities = ["video", "audio"]` | FL2VA | `t2va` | None |
| Text-to-video only | [`video_only.toml`](../examples/minimax_h3/video_only.toml): `target_modalities = ["video"]` | FL2VA | `t2va` | None |
| Text-to-audio only | [`audio_only.toml`](../examples/minimax_h3/audio_only.toml): `target_audio_directory` or `audio_jsonl_file`, with `target_modalities = ["audio"]` | FL2VA | `t2va` | None |
| Video-to-audio | [`av.toml`](../examples/minimax_h3/av.toml): `target_modalities = ["video", "audio"]` | FL2VA | `t2va` | `--h3_observed_modality video` |
| Audio-to-video | [`av.toml`](../examples/minimax_h3/av.toml): `target_modalities = ["video", "audio"]` | FL2VA | `t2va` | `--h3_observed_modality audio` |
| First-frame image-to-video+audio | [`i2va.toml`](../examples/minimax_h3/i2va.toml): video source; first frame comes from the target | FL2VA | `i2va` | None |
| First+last-frame-to-video+audio | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml): video source; keyframes come from the target | FL2VA | `fl2va` | None |
| Last-frame image-to-video+audio | [`l2va.toml`](../examples/minimax_h3/l2va.toml): video source; last frame comes from the target | FL2VA | `l2va` | None |
| Fixed arbitrary references | [`ref2va.toml`](../examples/minimax_h3/ref2va.toml): target video, [`ref2va_guides.toml`](../examples/minimax_h3/ref2va_guides.toml): target-derived timeline guides, [`ref2va_aligned_guide.toml`](../examples/minimax_h3/ref2va_aligned_guide.toml): external paired video guide, or [`image_ref2va.toml`](../examples/minimax_h3/image_ref2va.toml): target image; an audio target may carry references too | Ref2VA | `ref2va` | `--h3_training_mode ref2va`; use `source_*_directory`, optionally `aligned_guide_indices`, target-derived keyframes, or `--h3_guide_specs` |
| Zero-or-more arbitrary references | [`ref2va_omni.toml`](../examples/minimax_h3/ref2va_omni.toml): set `target_modalities` for the JSONL target; rows may omit references or use numbered `control_path_N` | Ref2VA | `ref2va_omni` | `--h3_training_mode ref2va_omni` |

For observed-modality objectives, the option names the modality supplied as clean **conditioning**, not the prediction target:
`--h3_observed_modality video` defines video-to-audio training. See [Training modes](#training-modes) for the corresponding
noise and loss contracts.

## Model download

Follow the repository's [installation instructions](../README.md#installation) first. Python 3.10–3.12.

All files come from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| File | Needed for |
| --- | --- |
| [`diffusion_models/minimax_h3_fl2va_bf16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors) | Everything except Ref2VA |
| [`diffusion_models/minimax_h3_ref2va_bf16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_bf16.safetensors) | Ref2VA |
| [`text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors) | Conditioning cache |
| [`vae/minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | Video latents |
| [`vae/minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | Audio latents |
| [`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | Optional pre-quantized conditioner |
| [`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | Optional pre-quantized transformer |
| [`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | Optional pre-quantized Ref2VA transformer |

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir /models/MiniMax-H3
```

Add any of the other files to the same command, for example the Ref2VA transformer:

```shell
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --local-dir /models/MiniMax-H3
```

```text
/models/MiniMax-H3/
├── diffusion_models/
│   ├── minimax_h3_fl2va_bf16.safetensors
│   ├── minimax_h3_fl2va_pruned_int8_convrot.safetensors  # optional
│   ├── minimax_h3_ref2va_bf16.safetensors                # Ref2VA only
│   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors # optional, Ref2VA only
├── text_encoders/
│   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      # optional
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
```

The tokenizer and processor metadata ship with the package, so `--tokenizer` is not needed. A checkpoint of the wrong variant is
rejected.

To list a checkpoint's components, shards, tensor and parameter counts without allocating tensor storage:

```shell
python minimax_h3_generate_video.py \
  --model /models/MiniMax-H3 \
  --prompt "unused" --output unused.mp4 \
  --inspect
```

## Train FL2VA

Use this workflow with `minimax_h3_fl2va_bf16.safetensors`. FL2VA covers text-only generation, target-derived first/last-frame conditioning, image targets, video targets, and audio targets. It does **not** accept arbitrary external media references; use [Train Ref2VA](#train-ref2va) for those.

### Choose an FL2VA task

The text-cache `--task` selects the conditioning presentation. The latent-cache command does not take `--task`.

| Goal | Text-cache `--task` | Additional cache arguments | Dataset example |
| --- | --- | --- | --- |
| Text → image, video, video+audio, or audio | `t2va` | None | [`image_fl2va.toml`](../examples/minimax_h3/image_fl2va.toml), [`t2va.toml`](../examples/minimax_h3/t2va.toml), [`audio_only.toml`](../examples/minimax_h3/audio_only.toml) |
| First image → edited image | `fl2va` | Both caches: `--h3_image_mode first` | [`image_fl2va_first.toml`](../examples/minimax_h3/image_fl2va_first.toml) |
| First and last images → edited image | `fl2va` | Both caches: `--h3_image_mode first_last` | [`image_fl2va_first_last.toml`](../examples/minimax_h3/image_fl2va_first_last.toml) |
| First frame → video+audio | `i2va` | None | [`i2va.toml`](../examples/minimax_h3/i2va.toml) |
| First and last frames → video+audio | `fl2va` | None | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml) |
| Last frame → video+audio | `l2va` | None | [`l2va.toml`](../examples/minimax_h3/l2va.toml) |

The image-editing rows use separate basename-matched source images and require the shown `--h3_image_mode` on both cache commands.
The video `i2va`, `fl2va`, and `l2va` rows take their keyframes from the target itself. They are not Ref2VA reference directories.
Use the selected task for text caching and the resulting cache in training; incompatible caches are rejected.

### Prepare an FL2VA dataset

Create `dataset.toml`. This minimal example trains text-to-video with the video's embedded audio:

```toml
[general]
resolution = [1344, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
target_video_directory = "/data/targets"
target_modalities = ["video", "audio"]
cache_directory = "/data/cache/fl2va"
source_fps = 24.0
target_frames = [124]
frame_extraction = "uniform"
```

Place `clip.mp4` and `clip.txt` together in `/data/targets`. For video-only training use `target_modalities = ["video"]`. For a separate synchronized soundtrack, also set `target_audio_directory`; files are matched by basename. Image and audio-only layouts are shown in the linked examples. H3 video frame counts must follow the `17k+5` grid.

### Cache FL2VA latents

```shell
python minimax_h3_cache_latents.py \
  --dataset_config dataset.toml \
  --vae /models/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --device cuda \
  --skip_existing
```

- Keep `--audio_vae` when any target or reference includes audio; omit it for image-only and video-only data.
- Omit `--vae` only for audio-only targets with no visual references.
- `--skip_existing` validates and reuses compatible caches. On a large, unchanged dataset add `--faster_check`; it fully validates `--faster_check_samples` items (default 8) first, but cannot detect a source file replaced in place.

### Cache FL2VA text embeddings

Set `--task` to the value chosen in [Choose an FL2VA task](#choose-an-fl2va-task). This example uses `t2va`:

```shell
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq \
  --h3_text_encoder_blocks_to_stream 50 \
  --task t2va --device cuda \
  --skip_existing
```

The native NVFP4/AWQ checkpoint reduces conditioner weight residency. If only the BF16 checkpoint is available, use
`--text_encoder_quantization nvfp4` to quantize its Linear weights to block-16 NVFP4 while loading; activations and GEMMs stay
BF16, so this W4A16 path also works on pre-Blackwell CUDA GPUs. `--h3_text_encoder_blocks_to_stream 50` streams all 50 frozen
language blocks and minimizes their simultaneous GPU residency; smaller values reduce transfer overhead but retain more blocks on
the GPU. Streaming works with BF16 and both NVFP4 modes, but not with the bitsandbytes `int8` or `nf4` loaders. If the full BF16
conditioner fits, use `--text_encoder_quantization none` and omit streaming to avoid quantization and block-transfer overhead. Add
`--cache_guidance_empty` if training will use caption dropout or guidance distillation.

### Start FL2VA training

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --blocks_to_swap 40 --block_swap_h2d_only --block_swap_ring_size 2 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --save_state --autoresume \
  --output_dir output --output_name h3_fl2va \
  --logging_dir logs --log_with tensorboard --log_grad_metrics
```

The example swaps 40 of the 50 main blocks. Increase `--blocks_to_swap` if model-weight residency still exceeds available VRAM, or remove all three block-swap options when the frozen base fits. Block swapping transfers weights and therefore reduces throughput. `--block_swap_h2d_only` is valid here because ordinary LoRA keeps the base frozen; the shown `--gradient_checkpointing` is required with this mode during training. Do not enable pinned block-swap memory unless the host has enough free RAM for the pinned block buffers. See [Memory and speed](#memory-and-speed) for quantization and kernel options.

**Next.** Track the run with [Validation](#validation) and, once a checkpoint exists, generate with [Inference](#inference); FL2VA also supports [sampling during training](#sampling-during-fl2va-training). [Memory and speed](#memory-and-speed) lists what to reach for when the run does not fit or is too slow.

## Train Ref2VA

Use this workflow with `minimax_h3_ref2va_bf16.safetensors`. Ref2VA adds clean image, video, and audio references to the conditioning side. Targets may independently be image, image+audio, video, video+audio, or audio. Different source/target mappings can coexist as separate `[[datasets]]` entries in one TOML.

### Prepare a Ref2VA dataset

This example trains image-reference → video+audio. Files in the target and source directories are matched by basename:

```toml
[general]
resolution = [1344, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
target_video_directory = "/data/targets"
target_modalities = ["video", "audio"]
source_image_directory = "/data/references"
source_modalities = ["image"]
cache_directory = "/data/cache/ref2va"
source_fps = 24.0
target_frames = [124]
frame_extraction = "uniform"
```

For example, `/data/targets/clip.mp4`, `/data/targets/clip.txt`, and `/data/references/clip.png` form one item. Use `source_video_directory` and/or `source_audio_directory` for other reference modalities. Set `source_video_audio_embedded = true` when each source video's own soundtrack is part of the reference, or `source_video_audio_paired = true` when separate video and audio source directories form one synchronized AV reference. See [`ref2va_modality_matrix.toml`](../examples/minimax_h3/ref2va_modality_matrix.toml) for image, video, AV, and audio targets in one config.

Use `--task ref2va` when every record must have references. Use `ref2va_omni` only when records may intentionally have zero references. Audio-only references are supported for training but experimental because released inference requires reference audio to accompany an image or video.

### Cache Ref2VA latents

```shell
python minimax_h3_cache_latents.py \
  --dataset_config dataset.toml \
  --vae /models/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --device cuda \
  --skip_existing
```

The VAE encodes both targets and DiT-side references. Keep `--audio_vae` if a target or reference has audio. Keep `--vae` even for an audio-only target when it has an image or video reference. Reference-video sizing options change cache identity. Pass the same `--reference_video_short_edge` and `--reference_video_max_pixels` values to both cache commands, training, and inference. Pass the same `--reference_video_fps` value to both cache commands and training; inference has no `--reference_video_fps` option.

### Cache Ref2VA text embeddings

```shell
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq \
  --h3_text_encoder_blocks_to_stream 50 \
  --task ref2va --device cuda \
  --skip_existing
```

The memory trade-off stated in [Cache FL2VA text embeddings](#cache-fl2va-text-embeddings) applies here too: native NVFP4/AWQ plus 50 streamed blocks minimizes simultaneous conditioner weight residency. BF16 without streaming avoids quantization and block transfers when it fits. Do not combine block streaming with bitsandbytes `int8` or `nf4`. If `source_modality_probabilities` is configured, the text cache stores its enabled reference presentations; changing those probabilities requires compatible text caches and is detected by `--skip_existing`.

### Start Ref2VA training

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --h3_training_mode ref2va \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --blocks_to_swap 40 --block_swap_h2d_only --block_swap_ring_size 2 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --save_state --autoresume \
  --output_dir output --output_name h3_ref2va \
  --logging_dir logs --log_with tensorboard --log_grad_metrics
```

For a `ref2va_omni` text cache, change the trainer option to `--h3_training_mode ref2va_omni`. Ref2VA reference rows lengthen the packed sequence, so it normally needs more activation memory than an otherwise identical FL2VA run. Reduce reference resolution or frame sampling first, then increase block swap if model-weight residency is still the problem; block swapping does not reduce attention activations.

**Next.** Track the run with [Validation](#validation) and, once a checkpoint exists, generate with [Inference](#inference), which is the only way to sample Ref2VA. [Memory and speed](#memory-and-speed) lists what to reach for when the run does not fit or is too slow.

## Dataset

H3 uses the explicit source/target schema below. A target video's embedded soundtrack is the audio target unless `target_audio_directory` supplies a separate basename-matched track; a video with
no audio stream trains as video-only with its audio loss masked. `source_*_directory` fields hold references whose basename matches
each target: controls for target `X` are `X.<ext>` or `X_<n>.<ext>`. A target whose own name ends in `_<n>` and that has no
direct match falls back to the shared prefix, but only over controls no other target claims; contested files are an error.
Relative `control_path`, `control_path_N`, `control_video_path_N`, and `control_audio_path_N`
values in a JSONL always resolve against that JSONL's directory. To combine separately stored video and audio as one
synchronized AV reference, use matching directories:

```toml
[[datasets]]
target_video_directory = "/data/targets"
target_modalities = ["video", "audio"]
source_video_directory = "/data/reference_video"
source_audio_directory = "/data/reference_audio"
source_modalities = ["video", "audio"]
source_video_audio_paired = true
cache_directory = "/data/cache"
target_frames = [39]
```

The three directories must contain basename-matched files. To condition on video without audio, declare only
`source_video_directory`; to condition only on audio, declare only `source_audio_directory`. Declare both and set
`source_video_audio_paired = true` to combine them into one synchronized AV reference. To use the soundtrack embedded in each
source video instead, declare `source_video_directory`, set `source_modalities = ["video", "audio"]`, and set
`source_video_audio_embedded = true`; use `source_modalities = ["audio"]` instead to retain only that embedded soundtrack. Do
not also declare `source_audio_directory`. A source directory may contain numbered
same-modality references such as `X_0.png` and `X_1.png`. An audio-only reference set — a voice clip and a caption, with no image or video reference — is legal for training
(**experimental**): the released inference distribution always pairs reference audio with an image or video, so a LoRA trained
this way runs off the base model's reference statistics and needs careful validation. Inference guards are unchanged:
`--reference_audio` still requires `--reference_image` or `--reference_video`.

Both caches record which reference files produced them, so swapping, reordering, or editing a reference rebuilds that item under
`--skip_existing`.

A dataset may read its latents from another dataset's cache: `latent_cache_directory` points at that directory while
`cache_directory` keeps the dataset's own text-encoder outputs. It exists for rollout-teacher datasets that share the
student's clips; see [Building the teacher dataset](#building-the-teacher-dataset).

### External aligned video guides

To train a LoRA with an external video aligned to the target timeline, select one or more source-video references with
`aligned_guide_indices`. The selected video is sampled on H3's 24-fps timeline, padded or cropped to the target duration,
and resized to the target bucket before VAE encoding. Its latent rows are packed as clean video conditioning with sigma
`0.999`; they are not included in the loss. The guide is omitted from the Qwen3-VL input, so it contributes only DiT-side
video rows.

```toml
[[datasets]]
target_video_directory = "/data/targets"
target_modalities = ["video"]
source_image_directory = "/data/reference_images" # reference index 0: ordinary Ref2VA reference
source_video_directory = "/data/aligned_guides"   # reference index 1: aligned guide
source_modalities = ["image", "video"]
aligned_guide_indices = [1]
cache_directory = "/data/cache"
target_frames = [124]
```

For directory datasets, indices follow the packed source order: images, then videos, then audio; numbered basename matches
retain their order within each modality. For JSONL datasets, indices follow `control_path_N` order. An aligned entry must
select a visual-only video; its soundtrack is ignored. After VAE encoding, its `(frames, height, width)` latent geometry must
equal the target latent geometry exactly. Re-cache both latents and text outputs after changing the list.
`source_modality_probabilities` is incompatible with aligned guides because the latent and text cache presentations must carry
the same aligned-guide count.

Ordinary video references start at the current reference-media RoPE time and advance that clock. An aligned guide does not
advance the reference clock. Instead, every guide latent at `(t, y, x)` receives the same three-dimensional position ID as the
target latent at `(t, y, x)`. Token values, token tags, conditioning timesteps, and the attention graph are otherwise unchanged.
Relative to packing the same video as an ordinary reference, this removes only the positional offset between the two streams;
it does not add a loss term or expose additional target data. Measure its effect against an ordinary-reference run with the
same data and training parameters. The equivalent inference layout is a full-length guide beginning at pixel frame zero:
`--guide_video 0:/path/to/guide.mp4`.

For modality dropout, provide probabilities in `[av, video, audio]` order:

```toml
source_modality_probabilities = [0.5, 0.25, 0.25]
```

The text cache stores every enabled presentation and training draws one mode per item. The same draw is reused by the trainable,
guidance, and preservation forwards. Video-only removes reference soundtracks; audio-only retains image references as visual
anchors and uses the audio from paired or embedded AV references. Recache text outputs after changing the probabilities. Under `--skip_existing`
the change is detected automatically and the affected caches are rebuilt. Latent caches do not
need to be rebuilt. Every enabled mode must leave at least one reference of some kind, checked when the dataset config is parsed:
a nonzero `video` weight needs an image or video reference. A nonzero `audio` weight needs an image, an audio source, or a video
explicitly declared as paired or embedded AV; a plain `source_video_directory` is visual-only. Audio-only survivors are permitted; the
off-distribution limitation above applies to them. On a reference set that is already audio-only the `audio` mode is the identity — it selects exactly
the same references as `av` — so dropout between those two weights has no effect there.

Reference audio is cropped or zero-padded to a canonical sample count (`temporal_shape`): audio paired with a video reference is
sized to that video's span, while a standalone audio reference is sized to the target's frame count. Reference rows carry no
validity mask, so padding added to a short reference track is
indistinguishable from silence to the model; keep reference audio at least as long as its reference video.

### Qwen control visuals

`qwen_control_directory`, `qwen_control_path`, and `qwen_control_path_N` attach control imagery — pose, depth, edges, sketch —
shown to the Qwen3-VL conditioner as visual context. Matching follows the source-directory rule (`X.<ext>` or `X_<n>.<ext>`);
relative JSONL paths resolve against the JSONL's directory. Images and videos only; at most 9 images and 3 videos per item.

```toml
[[datasets]]
target_video_directory = "/data/targets"
target_modalities = ["video", "audio"]
qwen_control_directory = "/data/pose"
cache_directory = "/data/cache"
target_frames = [39]
```

```json
{"video_path": "clip.mp4", "caption": "a dancer", "qwen_control_path_0": "pose/clip.mp4", "qwen_control_path_1": "depth/clip.png"}
```

| Field | Who sees it | Cost |
| --- | --- | --- |
| `source_*_directory` or JSONL `control_path_N` | pixel-space Ref2VA references: encoded by the VAE and packed as extra **reference rows in the DiT** | latent cache + longer packed sequence |
| `qwen_control_path`, `qwen_control_directory`, … | **only the text conditioner**; carried inside the Qwen presentation as vision-span tokens | text cache only; DiT input layout untouched |

Control spans close the visual prefix — after any keyframes or references, before the caption — continuing the existing
`<Picture N>` / `<Video N>` numbering. `--h3_text_visual_max_pixels` caps their size; control videos sample at
`--reference_video_fps` when set, otherwise at 2 fps, with no VAE preparation or soundtrack, within the 32768-token budget.
Every task accepts them, `t2va` included, and they compose with real Ref2VA references. The text cache fingerprints the control
files (`qwen_control_fingerprint`): any change rebuilds it under `--skip_existing`; latent caches are unaffected.

For CFG-style control dropout, cache with `--h3_qwen_control_dropout` — every item that carries controls also stores a
control-free presentation — and train with `--h3_qwen_control_dropout_rate P`, one synchronized draw per step that feeds the
same presentation to the guidance and base-preservation branches. The twin doubles the text-encoding work and the cached
presentations for those items; a rate above 0 refuses a cache written without the flag.

Experimental: the released H3 never saw control imagery in this channel, so verify control adherence against a prompt-only
baseline before relying on the recipe.

To restrict video or image loss to selected regions, set `loss_mask_directory` to a directory of masks with matching target
basenames, or set `default_loss_mask_path` as a fallback. A JSONL item may override either with `loss_mask_path` (the alias
`video_loss_mask_path` is also accepted). A mask may be a still image, video, or frame directory; still images repeat across the
clip and shorter mask sequences repeat their last frame. White pixels contribute at full strength, black pixels do not,
and grayscale values provide continuous weights between 0 and 1. The weighted loss is normalized by the sum of mask
weights, so softening a mask changes the spatial emphasis without also changing the effective learning rate.
During latent caching, `--h3_loss_mask_pooling max|average|nearest` controls how pixel weights are reduced to the H3 latent
grid. `max` preserves small selected regions, `average` preserves their fractional coverage, and `nearest` samples without
mixing neighboring values. Training defaults to `--h3_loss_mask_normalization weighted`; select `full` to divide by the full
latent element count so reducing mask coverage also proportionally reduces total gradient strength. Changing the pooling
mode invalidates matching masked latent caches automatically.
Mask images with alpha use that channel automatically; `loss_mask_use_alpha = true` instead uses a target image's alpha when no
external mask is available. `loss_mask_invert = true` reverses the mask. Masks are aligned
with the target crop during caching and pooled over H3's 5/17-frame latent windows.

```toml
[general]
resolution = [1344, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
target_video_directory = "/path/to/videos"
target_modalities = ["video", "audio"]
cache_directory = "/path/to/cache"
target_frames = [124, 175, 243, 294, 362]
frame_extraction = "uniform"
```

`target_modalities` selects which modalities are packed and scored:

| Value | Effect |
| --- | --- |
| `["image"]` | one-frame visual target |
| `["image", "audio"]` | one-frame visual target plus basename-matched `target_audio_directory`; uses `target_frames` for audio duration |
| `["video", "audio"]` | video and embedded audio, or basename-matched `target_audio_directory` |
| `["video"]` | video only; omits audio decoding, caching, and rows |
| `["audio"]` | audio only; needs `target_audio_directory` or `audio_jsonl_file`, and reuses `target_frames` for duration |

An audio-only dataset needs **exactly one** `target_frames` value, and it must be on the `17k+5` grid — the multi-value list in
the example above is video-only. `target_audio_directory` takes same-stem `.txt` captions; `audio_jsonl_file` takes records with
`audio_path` and `caption`.

An audio target may also declare Ref2VA references with `source_image_directory`, `source_video_directory`, and/or
`source_audio_directory`, or with per-record `control_path_N` / `control_video_path_N` +
`control_audio_path_N` / `control_modality_N` in the audio JSONL. That trains video-to-audio with an **arbitrary** conditioning
video (Foley), or audio generation from a reference voice clip plus a visual anchor. Reference composition follows the rule
above. Cache with `--task ref2va` and train with
`--h3_training_mode ref2va`; the packed sequence is `[text | references | target audio]` and no target video rows are emitted.
Reference video length follows the target's `target_frames`, and the RoPE spatial grid comes from the dataset `resolution` (the
same geometry an audio-only cache already records). With visual references, `--vae` is then required for latent
caching even though the target has no video. Audio-target caches are named `<stem>_audio<hash>_…`, where the hash covers the absolute source
path, so an audio file never overwrites a video cache of the same stem and two same-stem audio files from different directories
stay apart in one `cache_directory`.

By default, audio-only training emits no spatial target rows. Add `--h3_audio_only_spatial_tokens` to insert one
zero-initialized spatial DiT token per latent frame and disable loss on those tokens. They still participate in joint attention,
so the packed sequence contains spatial rows without introducing a video target. This changes the training layout and is opt-in;
it is not a memory switch. No video or cache rebuild is needed; temporal length is
derived from the cached audio and `target_frames` geometry.

For an audio-specialized LoRA, include the audio input/output projections as well as attention:

```shell
accelerate launch minimax_h3_train_network.py ... \
  --h3_audio_only_spatial_tokens \
  --h3_lora_targets "attention;audio"
```

The ordinary H3 LoRA default covers block attention and MLP projections but not the top-level `audio_patch_proj` and
`final_layer.audio_out`. Add the `audio` target when those input/output projections must be trainable. Selecting only `audio`
produces an endpoint-only adapter. `attention;audio` additionally trains cross-token attention while leaving the MLP projections
frozen.

Ref2VA reference and target modalities can be mixed across separate `[[datasets]]` entries in one config; see
[`ref2va_modality_matrix.toml`](../examples/minimax_h3/ref2va_modality_matrix.toml). The supported matrix is:

| Reference conditioning | Image | Image+audio | Video | Video+audio | Audio |
| --- | --- | --- | --- | --- | --- |
| image | yes | yes | yes | yes | yes |
| video without soundtrack | yes | yes | yes | yes | yes |
| audio only | experimental | experimental | experimental | experimental | experimental |
| image + audio | yes | yes | yes | yes | yes |
| paired video + audio | yes | yes | yes | yes | yes |

This is bidirectional across H3's media spaces: any image/video/audio conditioning subset (or text alone under T2VA/Ref2VA
Omni) can train any target column, and different mappings may share one run as separate `[[datasets]]` entries. H3 has one
visual output stream, so image and video are alternative visual target carriers rather than two simultaneous output heads; an
image target uses the one-frame visual latent path. Text remains conditioning rather than a predicted media target.

The target carrier is not arbitrary: image and video are alternative visual streams, while audio may accompany either or stand
alone. “Yes” means the cache and packed training layout support the combination; it does not imply that every combination was
represented in the released checkpoint.
Audio-only reference conditioning carries the inference limitation stated earlier in this section.

The explicit TOML vocabulary is `source_image_directory`, `source_video_directory`, and `source_audio_directory` for
conditioning. Targets use `target_image_directory`, `target_video_directory`, or `target_audio_directory`; a visual target may
add a basename-matched `target_audio_directory` for separately stored synchronized audio. `source_modalities` must describe the
media produced by the configured source directories; embedded source-video audio may add `audio` or select only `audio`, as described above. `target_modalities` is `["image"]`, `["image", "audio"]`, `["video"]`,
`["video", "audio"]`, or `["audio"]`; it selects the packed target rows and loss. Separately stored source modalities are matched to each target by basename and packed in image, video,
audio order. Set `source_video_audio_paired = true` when separate video and audio source directories form synchronized AV
references rather than independent references, or `source_video_audio_embedded = true` when a source video supplies both its
visual stream and embedded soundtrack. `source_modality_probabilities = [av, video, audio]` controls source dropout.

The released processor uses a 768-pixel short edge with a 1344×768 area cap. Other 32-pixel-aligned sizes work but sit outside
the released canvas distribution.

## Pre-caching

Both caches are required before training; the trainer never loads a VAE or the conditioner to rebuild a missing one.

```shell
python minimax_h3_cache_latents.py \
  --dataset_config dataset.toml \
  --vae /models/MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --device cuda

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task t2va --device cuda
```

Omit `--audio_vae` for image-only or video-only datasets whose references carry no audio either; it stays required whenever a
reference does (paired AV, embedded AV, or an audio source directory). Omit `--vae` for audio-only datasets without
visual references. Add `--cache_guidance_empty` if you plan to
use caption dropout or the guidance objective.

`--task` must match how you intend to train: `t2va` (text only), `i2va` (first frame), `fl2va` (first+last), `l2va`
(last frame), `ref2va`, or `ref2va_omni`. Keyframe tasks take their frames from the target video itself, not from control fields.

`--skip_existing` opens every cache to confirm it was written for the current options, which means loading every source item. On
datasets of thousands of items, add `--faster_check` to recognize caches by filename instead; it validates
`--faster_check_samples` caches (default 8) in full first and falls back to checking everything if any of them no longer matches.
It cannot see a source file replaced in place or a cache truncated by an interrupted write.

For FL2VA conditioned-image training, pass the same `--h3_image_mode first` or `--h3_image_mode first_last` to both cache
commands and use `--task fl2va` for text caching. `h3_image_frame_count` in the dataset, or the CLI override
`--h3_image_frame_count`, selects a `17k+5` target grid and defaults to 5. A still target repeats across that grid;
`multiple_target = true` instead resamples an ordered target-image sequence. The latent and text caches carry the same
source fingerprint, so changed/reordered targets or controls are rebuilt under `--skip_existing` and mismatched cache pairs
are rejected during training. `--h3_text_visual_max_pixels` limits only the images presented to Qwen3-VL.
`--h3_max_caption_tokens N` optionally truncates only the caption while retaining every structural image/video token. Its
default `0` leaves captions unchanged; pass the same nonzero value to text caching and training.

A conditioned-image target may also use a basename-matched `target_audio_directory` with
`target_modalities = ["image", "audio"]`. Set its single `target_frames` value equal to `h3_image_frame_count` so the repeated
visual target and audio target share one temporal grid.

```shell
# First-frame I2V
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task i2va --device cuda

# First+last frame
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task fl2va --device cuda

# Last-frame L2V
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --task l2va --device cuda
```

> [!IMPORTANT]
> Regenerate both caches with the same checkout used for training. An `i2va`/`fl2va`/`l2va` run rejects a cache without keyframe rows
> rather than silently training as `t2va`.

To reduce conditioner VRAM, add `--text_encoder_quantization int8`, `nf4`, or `nvfp4` to the BF16 checkpoint. The `nvfp4` mode
performs calibration-free block-16 W4A16 quantization while loading and retains BF16 activations. Alternatively, load the released
pre-quantized file with `--text_encoder_quantization nvfp4_awq`:

```shell
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq \
  --task t2va --device cuda
```

They reduce GPU residency only: the BF16 checkpoint is still memory-mapped
into host address space while weights are converted, so the host RAM requirement is unchanged.

If the conditioner still does not fit, `--h3_text_encoder_blocks_to_stream N` keeps `N` of its 50 frozen language layers in
CPU memory and moves them through a fixed two-layer GPU ring during encoding. `50` minimizes weight residency; smaller values
trade less transfer overhead for less memory saved. This is opt-in, requires CUDA, and supports BF16, on-the-fly `nvfp4`, and
native `nvfp4_awq` checkpoints. It does not support the bitsandbytes `int8` or `nf4` loader.

On a Blackwell GPU with PyTorch 2.10 or newer, `--h3_nvfp4_scaled_mm` additionally quantizes Qwen activations to FP4 and
uses the hardware W4A4 matrix kernel. It applies only to a native `nvfp4_awq` conditioner and fails early on unsupported
hardware. Leave it off when conditioning fidelity matters most: the default path
keeps BF16 activations and uses NVFP4 only for stored weights.

## Training

The ordinary LoRA command comes first; the learned-context and slider sections below train different things and
can be skipped on a first read.
### Start LoRA training

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 --save_state --autoresume \
  --output_dir output --output_name h3_style \
  --logging_dir logs --log_with tensorboard --log_grad_metrics
```

For Ref2VA, swap the checkpoint and add `--h3_training_mode ref2va` (or `ref2va_omni`).

### Optimizers

`--optimizer_type` takes any musubi optimizer. AI Toolkit's Automagic v3 (**experimental**) is bundled as `Automagic3`:

```shell
accelerate launch minimax_h3_train_network.py ... \
  --optimizer_type Automagic3 --learning_rate 1e-4 --lr_scheduler constant --max_grad_norm 0
```

The dashboard's optimizer **Set** button applies these values. It adapts the rate itself, so `--learning_rate` is a starting
point and schedulers, warmup and decay are ignored; the adapted rate stays within two decades of it unless `min_lr`/`max_lr` are
passed in `--optimizer_args`. Weight decay defaults to `1e-4`. State costs about one byte per parameter.

Without target-selection options, adapters target attention and feed-forward projections; norms and timestep/modality
calibration stay frozen. H3 also provides named targeting so ordinary users do not need Python regular expressions:

| Target | Modules selected |
| --- | --- |
| `attention` | QKV and output projections in transformer attention |
| `mlp` | `fc1` and `fc2` in transformer MLPs |
| `audio` | `audio_patch_proj` and `final_layer.audio_out` |
| `video` | `video_patch_proj` and `final_layer.video_out` |
| `token_refiner` | the two text token-refiner blocks |

Pass all targets in one quoted expression, separated by semicolons. A name without a range selects the complete group:

```shell
--h3_lora_targets "attention;mlp;audio"
```

Add `:BLOCKS` to give attention and MLP independent block selections:

```shell
--h3_lora_targets "attention:0-7,24-31;mlp:8-15,32-49;audio"
```

H3 has blocks `0` through `49`. Only `attention` and `mlp` accept ranges; audio/video endpoints and token refiners are not
numbered main blocks. Named targets deliberately cannot be mixed with raw `include_patterns` or
`exclude_patterns`, which avoids ambiguous precedence. Advanced users may omit the named options and continue using the raw
regex network arguments; those patterns are matched against complete module paths with `fullmatch`, and an include pattern is
an exception to exclusions rather than a universal standalone allow-list.

NoRA (Normalized Low-Rank Adaptation, [Kang et al. 2026, arXiv:2608.31036](https://arxiv.org/abs/2608.31036)) is available
through `--network_args`: `nora=forward` keeps every column of `lora_down` at unit norm on each forward (the saved adapter is an
ordinary LoRA and needs no NoRA code to merge or infer), `nora=init` normalises once at initialisation, and `init=bimi` gives
`lora_down` a block-identity start; set `--network_alpha` equal to `--network_dim`. Default `nora=off` is bit-identical to a
run without the argument.

LoHa/LoKr are unsupported. Regional `torch.compile` covers all 50 main blocks and both text-refiner blocks; use `--compile` and optionally
`--compile_auto_cache_size_limit`, `--compile_fallback_to_eager`, or `--inductor_config KEY=VALUE ...`. GPU compilation requires
a working Triton installation; on Windows, install a `triton-windows` build compatible with the installed PyTorch and Python
versions.


### Train a learned context

A learned context is a small trainable sequence composed with Qwen conditioning. Qwen and the H3 transformer remain frozen, and
training writes one embedding file instead of a LoRA.

Use the ordinary dataset TOML and caches described in [Train FL2VA](#train-fl2va) or [Train Ref2VA](#train-ref2va). Put the shared
behavior or appearance in every target, but omit it from captions so the learned rows must carry it. Use varied scenes and keep
separate held-out prompts or references to detect memorization. Targets may be real, edited, generated, or mixed; their cache format
and training command are identical. Edit the paths in
[`dataset.toml`](../examples/minimax_h3/learned_context/dataset.toml).

#### Without an existing dataset

If you have no dataset, edit [`bootstrap.toml`](../examples/minimax_h3/learned_context/bootstrap.toml) and generate targets with
the base model. The generator writes scene-only `.txt` captions beside videos whose prompts also contain the learned concept:

```shell
python examples/minimax_h3/learned_context/generate_bootstrap_targets.py \
  --bootstrap_config examples/minimax_h3/learned_context/bootstrap.toml \
  --model /path/to/minimax_h3_fl2va_int8_convrot.safetensors --int8_convrot_base \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq --h3_text_encoder_blocks_to_stream 50 \
  --vae /path/to/minimax_h3_video_vae.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae.safetensors \
  --duration 5 --ratio 16:9 --steps 20 \
  --blocks_to_swap 30 --block_swap_h2d_only
```

Set `target_video_directory` in `dataset.toml` to the generated directory.

#### With an existing dataset

Put videos and matching `.txt` captions in one directory and set it as `target_video_directory` in `dataset.toml`. Captions describe
the scene but omit the learned concept. When targets do not contain usable audio, set `target_modalities = ["video"]` and use
`--h3_audio_loss_weight 0`.

#### Train

In both cases, create the latent and `--task t2va` text caches as described in [Pre-caching](#pre-caching). Then initialize a
reusable attribute from a concise semantic description and prepend it to each cached caption:

```shell
accelerate launch minimax_h3_train_learned_context.py \
  --dataset_config examples/minimax_h3/learned_context/dataset.toml \
  --dit /path/to/minimax_h3_fl2va.safetensors --h3_training_mode fl2va \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq --h3_text_encoder_blocks_to_stream 50 \
  --h3_learned_context_init_prompt "water rapidly rises and floods the entire scene" \
  --h3_learned_context_composition prepend \
  --h3_shift_video 12 --h3_shift_audio 3 \
  --h3_loss_balance modality --h3_video_loss_weight 1 --h3_audio_loss_weight 1 \
  --learning_rate 1e-3 --max_train_steps 1000 \
  --output_dir output --output_name water_context \
  --sdpa --mixed_precision bf16 --save_precision bf16 --gradient_checkpointing \
  --h3_convrot_int8 --h3_convrot_int8_fwd int8 --h3_convrot_int8_bwd int8 \
  --h3_adaln_rank 16 \
  --blocks_to_swap 30 --block_swap_h2d_only --block_swap_ring_size 2
```

Reduce `--blocks_to_swap` for faster training when more device memory is available.

Use the checkpoint and `--h3_training_mode` matching the cached task. Start around `1e-3` for a semantic initializer. Compare fixed
base and trained renders on held-out inputs; a lower training loss alone does not prove that the context transfers.

Exactly one initializer is required:

| Option | Behavior |
| --- | --- |
| `--h3_learned_context_init FILE` | Continue training from a saved learned embedding. |
| `--h3_learned_context_init_prompt TEXT --text_encoder FILE` | Start a new context by encoding `TEXT` once with frozen Qwen and training the resulting rows. |

`--h3_learned_context_init_prompt` initializes the single learned context shared by the whole training run. The trainer encodes
it once before loading H3 and then unloads Qwen; it is not a per-item caption. With multiple varied dataset items, use `prepend`
so every item retains its own cached caption alongside the shared learned context.

To continue a saved embedding, reuse the command with `--h3_learned_context_init output/water_context.safetensors` instead of
`--h3_learned_context_init_prompt` and its text-encoder options, and select a new `--output_name`.

`--h3_learned_context_composition prepend` retains each cached caption; `replace` drops it. Prepend when the item captions must stay available to the model, replace when the learned context is meant to stand alone.
`replace` discards the cached caption and trains a self-contained context. The saved file contains the optimized Qwen rows under
the `qwen3vl_32b` tensor key.

To train while a fixed LoRA is applied to H3, pass it with `--h3_overlay_weights`.

### Use a learned context

Pass the saved file to `minimax_h3_generate_video.py`:

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_fl2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "a person in a studio" --seed 42 \
  --h3_learned_context output/my_context.safetensors \
  --output context.mp4
```

Repeat `--h3_learned_context` to prepend multiple learned contexts in command-line order. Repeat
`--h3_learned_context_multiplier` to scale the corresponding files; omitted multipliers default to `1`, and `0` disables that
context. Learned contexts can be used together with `--lora_weight`.

The saved file also uses ComfyUI's H3 embedding format. Place it in a ComfyUI embeddings directory and use
`embedding:my_context` in the prompt.

### Train a slider LoRA

A slider LoRA learns opposite behavior at positive and negative adapter multipliers. H3 provides three opt-in objectives:

| Slider mode | Training signal | H3 family |
| --- | --- | --- |
| `text` | Frozen-model predictions for positive, neutral, and negative prompts | FL2VA |
| `reference` | Filename-matched positive and negative target caches with one shared text presentation | FL2VA |
| `ref2va` | Filename-matched targets with one separate shared Ref2VA conditioning cache | Ref2VA or Ref2VA Omni |

Copy and edit [`slider_text.toml`](../examples/minimax_h3/slider_text.toml) for a prompt-defined slider, or
[`slider_paired.toml`](../examples/minimax_h3/slider_paired.toml) for cached image, video, audio, or joint AV pairs. Then run:

```shell
accelerate launch minimax_h3_train_slider.py \
  --slider_config examples/minimax_h3/slider_text.toml \
  --dit /path/to/minimax_h3_fl2va_bf16.safetensors --h3_training_mode fl2va \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --network_module networks.lora_minimax_h3 --network_dim 16 --network_alpha 16 \
  --h3_base_preservation_loss_weight 0.02 --h3_base_preservation_probability 0.25 \
  --learning_rate 1e-4 --max_train_steps 500 \
  --output_dir output --output_name h3_slider \
  --sdpa --mixed_precision bf16 --gradient_checkpointing
```

`target_modality` is `video`, `audio`, or `av`. In text mode, `latent_frames = 1` selects the still-image route.
`latent_height` and `latent_width` are latent dimensions, so divide pixel dimensions by 16; both must be even because H3 uses
2×2 spatial patches.
Text prompts are encoded once before H3 is loaded. Paired modes consume the ordinary H3 latent and text cache files directly;
positive and negative files must have identical basenames and target shapes. For `ref2va`, `conditioning_cache_dir` supplies the
shared text and reference rows while the positive and negative directories supply only the targets.

Every text target requires `positive`, `negative`, and `target_class`. H3 derives the neutral teacher from a null-instruction
version of `target_class`, preserving its text-row count and the media rotary layout; `target_class` is also the prompt used for
the two gradient-bearing `+1` and `-1` passes.

The slider objective composes with the regular H3 guidance-distillation and base-preservation options. Paired-media sliders use
these options through the ordinary H3 velocity-loss path. Text sliders apply the same guidance formula to their teacher-prediction
target, which can increase the effective direction strength; start with base preservation alone unless that extrapolation is
intentional. The text null presentation is encoded online with the same text-row layout. `--h3_fuse_frozen_teachers` is not
available for text sliders.
Observed-modality, masking, extension, caption-dropout, and CREPA objectives remain unsupported. The trainer otherwise reuses the
normal H3 LoRA targeting, quantization, checkpointing, block-swap, optimizer, validation-sampling, and save options. Sampling during
training evaluates every value in `sample_slider_range`.

### Training a guidance-distilled model

The released H3 checkpoints are guidance-distilled: one prompted forward returns what an undistilled model would
produce with classifier-free guidance, i.e. the empty-prompt prediction plus a scale times the difference between the
prompted and the empty-prompt predictions. The flow-matching loss does not model this. Its minimum is the
un-amplified prediction, so fine-tuning with the plain loss removes the built-in amplification: the training loss
falls while prompt adherence, contrast and detail in the output degrade.

The methods below keep the frozen checkpoint in the objective so that the amplification survives. Each entry has a
short description, advantages, disadvantages with the step cost relative to a plain LoRA step, and the flags that
enable it. Metrics and the current recommended settings for the two checkpoints follow.

Terms. The *field* is the difference between the prompted and the empty-prompt prediction, the quantity the
amplification scales; *field length* is its norm, *field direction* its orientation. The *null branch* is the
empty-prompt prediction. Costs are approximate step-time ratios.

#### Methods

##### Plain LoRA

LoRA training with the unmodified flow-matching loss. Nothing references the frozen checkpoint; the field length
falls to a fraction of the checkpoint's within a few hundred steps.

- **Applies to:** FL2VA and Ref2VA alike.
- **Advantages:** lowest cost; fastest concept acquisition. Useful as a baseline, not as a deliverable.
- **Disadvantages:** removes the amplification. The output has visible artifacts (soft, low-contrast, incoherent
  frames) at any prompt, including the training resolution and length. Cost **1x** (the reference for the ratios
  below).
- **Flags:** none.

##### De-distilling adapter (`--base_weights`)

A separately trained adapter is merged into the frozen base for the duration of training, so the training base
approximates an undistilled model. The adapter exists only during training: at inference the LoRA goes onto the
stock checkpoint without the adapter and without an extra forward. Public examples for H3:
`ostris/minimax_h3_training_adapter` on Hugging Face and `DiffSynth-Studio/MiniMax-H3-TrainingAdapter` on
ModelScope (rank 64, used there through `--preset_lora_path`).

- **Applies to:** both checkpoints; an adapter is trained against one checkpoint and is used with that one.
- **Advantages:** no change to the training loop or to inference; combines with any objective.
- **Disadvantages:** the adapter is an approximation of the undistilled model and its error enters the LoRA.
  Cost about **1x**.
- **Flags:** `--base_weights adapter.safetensors` (repeatable), `--base_weights_multiplier`; optionally
  `--h3_base_preservation_loss_weight` and `--h3_base_preservation_probability`.

##### Contrastive guidance loss

Each step also evaluates the model with the empty prompt, and the target becomes the empty-prompt prediction plus S
times the offset from it to the data velocity. The empty-prompt prediction is the student's own and moves during
training.

- **Applies to:** both checkpoints, same flags.
- **Advantages:** one extra forward; no second dataset; keeps a useful part of the field length.
- **Disadvantages:** the reference point moves with the student, so the field can shorten or rotate while the loss
  is satisfied; S must be measured per checkpoint. Cost **1.4x to 1.6x**.
- **Flags:** `--h3_guidance_distillation_scale S`, `--h3_guidance_loss_form contrastive`; optionally
  `--h3_guidance_loss_schedule {sigma,constant}`, `--h3_guidance_distillation_probability`,
  `--h3_guidance_audio_scale`, `--h3_guidance_cfg_zero`, `--h3_guidance_scale_range`. Needs a text cache built with
  `--cache_guidance_empty`.

##### Frozen-null target with the null anchor

The contrastive target with a fixed reference point: the empty-prompt prediction comes from the frozen checkpoint,
the target is normalised against it, and an anchor term holds the student's empty-prompt prediction at the frozen
one.

- **Applies to:** both checkpoints, same flags; the anchor weight differs (1.0 with a reference condition, 0.3 with a
  trigger word).
- **Advantages:** the most effective single term for keeping the field; the anchor gradient is nearly orthogonal to
  the data gradient; what the adapter learns is conditional on the prompt.
- **Disadvantages:** with anchor weight 1.0 and a weak condition (a single trigger word) the concept is not
  learned; anchor 0.3 learns it partially. Cost about **1.8x** on its own.
- **Flags:** `--h3_guidance_loss_form normalized`, `--h3_guidance_null_source frozen`,
  `--h3_guidance_null_anchor_weight W`; optionally `--h3_guidance_null_anchor_weight_end` (linear ramp of the weight
  over training), `--h3_guidance_null_anchor_probability`, `--h3_guidance_loss_schedule constant`,
  `--h3_caption_dropout_rate 0` (dropout trains the branch the anchor holds), `--h3_fuse_frozen_teachers`.

##### Rollout supervision

The student runs its own sampler a few steps from noise and is supervised at the resulting states against the frozen
checkpoint given information the student does not receive: the clip's reference or keyframe, or a richer caption.
This is the on-policy objective of D-OPSD. Alone it is weaker than the mixtures below; in practice it runs on a
fraction of the steps and a guidance target runs on the rest.

- **Hybrid D-OPSD (H-OPSD):** rollout supervision on half the steps, the contrastive target on the other half. It
  learns a style at the base model's output quality. The style is unconditional: it also appears on prompts without
  the trigger word.
- **Applies to:** both checkpoints; what differs is the privilege. Ref2VA: `--h3_rollout_teacher_privilege reference`,
  the teacher receives more references than the student (frames of the target itself); the teacher forward runs over
  the reference rows, so rollout steps 2 and window 1 keep the cost down. FL2VA: `keyframe`, the teacher receives the
  clip's first and last frames while the student trains text-only (`t2va` cache); rollout steps 4 and window 2 are
  affordable. `caption` is a privilege only when the teacher caption carries information the student's does not.
- **Advantages:** supervises the states the sampler visits; corrects field direction; the teacher is the same
  checkpoint.
- **Disadvantages:** a second cached dataset for the teacher; the teacher must receive information the student
  does not (the trainer rejects a teacher whose conditioning the student also has); the teacher predicts with less
  amplification than the checkpoint, so the field still shortens unless the floor is added. Cost **1.9x** on FL2VA
  and **5x to 7x** on Ref2VA, where the teacher input includes reference frames.
- **Flags:** `--h3_rollout_supervision`, `--h3_rollout_teacher_config teacher.toml`,
  `--h3_rollout_teacher_privilege {auto,reference,keyframe,caption,qwen}`, `--h3_rollout_probability` (0.5),
  `--h3_rollout_steps` (2), `--h3_rollout_window` (1), `--h3_rollout_fused_teacher` (all frozen arms in one pass;
  use it); optionally `--h3_rollout_prefix`, `--h3_rollout_stop_min`, `--h3_rollout_stop_shifted`,
  `--h3_rollout_null_anchor_weight`.

###### Building the teacher dataset

Copy the training dataset config, add the privileged field, and give it its own `cache_directory`. On the reference
channel the teacher's source directory holds more references per clip than the student's, including frames from the
target itself. On the caption or keyframe channel the clips are the same, so the teacher reads the student's
latents and caches only its text presentation:

```toml
[[datasets]]
video_jsonl_file = "/data/train.jsonl"                 # the student's items
cache_directory = "/data/cache/fl2va_teacher"          # the teacher's text-encoder cache
latent_cache_directory = "/data/cache/fl2va"           # the student's latents, read as-is
target_frames = [124]
```

Cache the teacher config with the same two cache commands and the same sizing options as the training set. Under
`--skip_existing` the latent command skips existing files and never purges a borrowed directory. A keyframe teacher
needs a text cache made with the keyframe task (`--task fl2va` or `i2va`); a reference teacher caches both halves.

##### Field length floor

At the rollout states the length of the student's field is compared with the frozen checkpoint's at the same state
and the step is penalised when the student's is shorter. It does not act on the direction, and it is measured
against the frozen empty branch, so it is used together with the null anchor.

- **Applies to:** both checkpoints, same flags.
- **Advantages:** restores the length the rollout teacher removes; no extra forward under the fused teacher.
- **Disadvantages:** the weight matters (0.3 works, 1.0 prevents learning); without the anchor it produces artifacts;
  it raises `val/rollout/field` directly, so judge it by `val/rollout/field_cos`, `val/drift/prompted_rel` and
  renders. An upper bound on the same ratio (`--h3_rollout_field_cap`) is not recommended.
- **Flags:** `--h3_rollout_field_floor 0.3` (needs rollout supervision, the frozen null and an anchor); optionally
  `--h3_rollout_field_floor_sigma_max`, `--h3_rollout_field_floor_direction {self,teacher}`.
- **Direction-only teacher:** `--h3_rollout_teacher_magnitude_weight 0` makes the rollout term match the teacher's
  field direction only and leaves the length to the floor, so the two terms stop opposing each other. With a
  reference-privileged teacher (Ref2VA) it keeps a longer field at the same direction and is part of that recipe.
  With a keyframe teacher on a trigger-word style (FL2VA) it suppresses the style: the teacher's direction is the
  base's, and following it alone leaves nothing to learn. Do not use it there. Upstream musubi-tuner applies the
  same split to its off-policy teacher-matching loss (kohya-ss/musubi-tuner#1065, #1086).

##### NAFP: Null-Anchored Field Preservation

The frozen-null target with the anchor, rollout supervision against a privileged teacher, and the length floor,
together. The anchor fixes the reference point, the teacher sets the direction, the floor holds the length.

- **Applies to:** both checkpoints. Ref2VA: reference privilege, anchor 1.0, direction-only teacher term. FL2VA:
  keyframe privilege, anchor 0.3, 1000 steps.
- **Advantages:** the only recipe that constrains length, direction and the null branch at once; keeps a longer
  field than rollout supervision alone at the same output quality. What it learns is conditional on the prompt.
- **Disadvantages:** the highest cost, three to four forwards per step: about **2.2x** on FL2VA, **3.5x to 4x** on
  Ref2VA. The anchor limits how much of an unconditional change (a rendering style) can be learned: with a trigger
  word and anchor 0.3 the style comes out partial; with anchor 1.0 it does not come out. Reference conditioning
  takes anchor 1.0; a trigger word takes 0.3.
- **Flags:** the union of the three sections above; the exact set is under *Current recommended settings*.

##### Stability controls

Optional additions to any recipe above, on either checkpoint.

- `--h3_adapter_ema_decay 0.999` keeps an EMA of the adapter, saved beside each checkpoint as
  `<name>-ema-step<N>.safetensors`; `--h3_validate_ema` validates the EMA instead of the live weights. Free.
- `--h3_adapter_prompt_only` runs every empty-prompt forward with the adapter disabled, which makes the null anchor
  redundant by construction. Weaker than the anchor: the shared weights still drift.
- `--h3_guidance_scale_sigma_max 0.9` disables the target amplification above that sigma, where the implied scale is
  very large and the data is close to noise. Improves validation fit; combine with the floor.
- `--h3_measured_variance_weighting curve.json` weights samples by a measured dispersion curve. Degrades field
  direction; not recommended.
- `--h3_validation_multipliers 0.75,1.25` validates at several LoRA multipliers in one pass.
- `--h3_validation_bare_dataset_config bare.toml`: the trigger probe. A second validation set over the same items
  with the trigger removed from the captions (latents shared through `latent_cache_directory`, text cached
  separately). Reports whether the adapter changes the prediction more with the trigger than without, i.e. whether
  the concept was learned conditionally. LoRA runs only; doubles the probe cost.
- Diagnostics: `--h3_term_grad_every N` (per-term gradient norms and cosines; single process, `--blocks_to_swap 0`),
  `--h3_adapter_stats_every N`, `--h3_train_sigma_bins`, `--h3_validation_std`.

#### Metrics

The training loss does not show the loss of amplification. Three validation metrics do, computed on the held-out set
every `--validate_every_n_steps` without rendering:

- `val/drift/prompted_rel`: how far the prompted prediction moved from the base model's, relative to the base's field
  length. The empty-prompt branch cancels in it, so no preservation term can improve it artificially. Below about
  0.8 the output shows no visible damage; around 1 and above it shows artifacts; the range between depends on the
  recipe (an anchored adapter reaches artifacts at a lower value than an unanchored one).
- `val/rollout/field` and `val/rollout/field_cos`: field length and direction at the states the adapted model's own
  sampler reaches. A plain LoRA drops the length to roughly a third of the base's within a few hundred steps. The
  floor raises the length directly, so judge it by the cosine.
- `val/velocity_err_rel`: the adapted model's prediction error over the base model's on the same targets. Below 1
  means the adapter learned something; it is the counterweight to the two metrics above, which an adapter that
  barely changed would pass.

Training metrics are enabled by the flags listed; everything under `val/` needs `--validation_dataset_config`, and
the field metrics need `--h3_validation_field_probe` or `--h3_validation_rollout_probe`.

| Metric | Enabled by | Meaning | Expected |
| --- | --- | --- | --- |
| `loss/video`, `loss/audio` | always | Training loss per modality under the active objective. | decreasing |
| `loss/video/bin{k}` | `--h3_train_sigma_bins` | Video loss by sigma bin (cuts 0.5 / 0.8 / 0.9). | top bin noisier |
| `loss/rollout_video` | rollout | Teacher term at the rollout states (direction part when the split is on). | decreasing |
| `loss/rollout_field_floor`, `h3/rollout_field_ratio` | floor | Floor penalty; student field length over the base's at rollout states. | ratio near 1 |
| `loss/rollout_teacher_magnitude`, `h3/rollout_teacher_length_ratio` | magnitude weight below 1 | Length gap to the teacher; student over teacher field length. | — |
| `h3/rollout_active`, `h3/rollout_stop_sigma` | rollout | Whether the step used the rollout objective; the stop sigma of the prefix. | mean of the first equals the probability |
| `h3/rollout_teacher_cos_base`, `h3/rollout_teacher_vs_base` | floor | How much the privileged teacher differs from the base at the rollout states. | below 1; 1 means the teacher adds nothing |
| `h3/null_anchor_weight` | anchor ramp | The anchor weight at this step. | follows the ramp |
| `h3/grad/<term>_norm`, `h3/grad/cos_<a>_<b>` | `--h3_term_grad_every` | Per-term gradient norms and pairwise cosines. | anchor cosine near 0; floor cosine negative |
| `h3/adapter/delta_norm`, `h3/adapter/ema_rel_dist` | `--h3_adapter_stats_every` | Norm of the adapter delta; distance of the live weights from the EMA. | delta increasing; distance below 0.3 |
| `val/loss` | validation | Held-out loss. | decreasing |
| `val/field`, `val/field_cos`, `val/field_dist` | field probe | Field length over the base's, cosine to the base's field, and a combined distance, at noised data states. | length near 1, cosine near 1, distance near 0 |
| `val/velocity_err_rel` | field probe | Adapted error over the base's error on the same target. | below 1 |
| `val/drift/prompted_rel` | field probe | Distance of the prompted prediction from the base's, relative to the base's field length. | below 0.8; near 1 means collapse |
| `val/drift/empty` | field probe | Distance of the empty-prompt prediction from the base's. | near 0 with an anchor |
| `val/trigger/drift_gain`, `val/trigger/err_gain` | bare dataset config | Prompted drift with the trigger minus without; fit improvement with the trigger minus without. | above 0 when the concept is conditional; zero or negative when the change is unconditional |
| `val/trigger/drift_gain/bin{k}`, `val/trigger/drift_bare`, `val/trigger/err_rel_bare`, `val/trigger/pairs` | bare dataset config | The same per sigma bin; the bare-caption values; the number of paired items. | — |
| `val/rollout/field`, `val/rollout/field_cos` | rollout probe | Field length and direction at the states the adapted sampler reaches. | length above 0.5, cosine near 1; do not judge the floor by the length alone |
| `val/rollout/x0_err`, `/step{i}` | rollout probe | Clean-clip error along the adapted model's own sampling trajectory; growth with `i` means compounding error. | flat |
| `loss/guidance_null_anchor`, `loss/rollout_null_anchor` | anchor weights above 0 | Data-state and rollout-state null anchor terms. | small, stable |
| `loss/base_preservation`, `loss/dop` | `--h3_base_preservation_loss_weight`, `--h3_dop_loss_weight` | Base-preservation and Differential Output Preservation terms. | — |
| `val/field_if_null_pinned` | field probe | Field length the adapter would have if its empty branch were held at the base's. | diagnostic only |
| `h3/sigma_video`, `h3/sigma_audio` | always | Mean shifted sigma of the step's draw per modality. | matches the schedule |
| `h3/*_active`, `h3/caption_dropped`, `h3/no_active_target` | the corresponding flag | 1.0 on steps where that path fired (guidance branch, base preservation, DOP, null anchor, rollout, caption or control dropout); `no_active_target` flags a step with nothing to supervise. | means track the configured probabilities; `no_active_target` never |
| `*_std` | `--h3_validation_std` | Per-clip standard deviation of the pooled metric. | — |

Checkpoint selection: take the step at which `val/drift/prompted_rel` is still below 0.8 and
`val/rollout/field_cos` is still high while `val/velocity_err_rel` has stopped improving, then confirm on renders.
Training past that point adds memorisation of the training clips, not concept quality.

#### Current recommended settings

The recipes list the parameters that keep training stable; the number of steps is not one of them. It depends on
the dataset: validate every 250 steps and stop by the checkpoint rule in *Metrics*. All recipes share the base
flags:

```shell
--network_module networks.lora_minimax_h3 --network_dim 16 --network_alpha 16 --h3_adaln_rank 16 \
--optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing --mixed_precision bf16 \
--timestep_sampling sigmoid --sdpa --blocks_to_swap 12 --block_swap_h2d_only --seed 1234 \
--validation_dataset_config holdout.toml --validate_every_n_steps 250 --validate_at_start \
--save_every_n_steps 250 --h3_validation_field_probe --h3_validation_rollout_probe 4
```

**Ref2VA: NAFP with the direction-only teacher.** About 3.5x to 4x a plain step. The recipe converges in fewer
steps than a plain LoRA and later checkpoints lose field direction without improving the fit, so the run is
shorter than a plain run on the same data. `--h3_rollout_teacher_magnitude_weight 0` keeps a longer field;
without it the recipe is the plain floor. Evaluating the null anchor on a fraction of the steps
(`--h3_guidance_null_anchor_probability`) saves step time but weakens the learned concept; keep it at 1.

```shell
--dit minimax_h3_ref2va_bf16.safetensors --h3_training_mode ref2va --reference_image_short_edge 384 \
--h3_rollout_supervision --h3_rollout_teacher_config teacher.toml --h3_rollout_teacher_privilege reference \
--h3_rollout_probability 0.5 --h3_rollout_steps 2 --h3_rollout_window 1 --h3_rollout_fused_teacher \
--h3_guidance_distillation_scale 5.0 --h3_guidance_loss_schedule constant --h3_guidance_loss_form normalized \
--h3_guidance_null_source frozen --h3_guidance_null_anchor_weight 1.0 --h3_caption_dropout_rate 0 \
--h3_rollout_field_floor 0.3 --h3_rollout_teacher_magnitude_weight 0
```

*Faster alternative:* H-OPSD with the reference-privileged teacher, about 3x a plain step. Output quality is the
same; the concept comes out weaker and the transformation of the subject less complete, and it is unconditional.
Take it when step time matters more than concept strength.

```shell
--dit minimax_h3_ref2va_bf16.safetensors --h3_training_mode ref2va --reference_image_short_edge 384 \
--h3_rollout_supervision --h3_rollout_teacher_config teacher.toml --h3_rollout_teacher_privilege reference \
--h3_rollout_probability 0.5 --h3_rollout_steps 2 --h3_rollout_window 1 --h3_rollout_fused_teacher \
--h3_guidance_distillation_scale 3.5 --h3_guidance_loss_form contrastive
```

**FL2VA, a style that applies to every prompt: H-OPSD.** About 2x a plain step. Full style at the base model's
output quality; the field is shorter than the base's and the style also appears on prompts without the trigger. The
teacher config uses the keyframe task (`--task fl2va` text cache); the student trains text-only (`t2va`).

```shell
--dit minimax_h3_fl2va_bf16.safetensors \
--h3_rollout_supervision --h3_rollout_teacher_config teacher.toml --h3_rollout_teacher_privilege keyframe \
--h3_rollout_probability 0.5 --h3_rollout_steps 4 --h3_rollout_window 2 --h3_rollout_fused_teacher \
--h3_guidance_distillation_scale 3.5 --h3_guidance_loss_form contrastive
```

*Faster alternative:* pure D-OPSD, the same flags without the two guidance flags, about 1.7x a plain step. Fit,
drift, field length and conditionality come out the same; the style is slightly weaker.

**FL2VA, a style switched by a trigger word: NAFP with anchor 0.3.** About 2.2x a plain step. Less style than
H-OPSD, output quality at the base model's, the field preserved, the style absent from prompts without the trigger.
Training past the point where the style has settled adds memorisation and artifacts, and a higher LoRA multiplier
at inference adds artifacts rather than style.

```shell
--dit minimax_h3_fl2va_bf16.safetensors \
--h3_rollout_supervision --h3_rollout_teacher_config teacher.toml --h3_rollout_teacher_privilege keyframe \
--h3_rollout_probability 0.5 --h3_rollout_steps 2 --h3_rollout_window 1 --h3_rollout_fused_teacher \
--h3_guidance_distillation_scale 5.0 --h3_guidance_loss_schedule constant --h3_guidance_loss_form normalized \
--h3_guidance_null_source frozen --h3_guidance_null_anchor_weight 0.3 --h3_caption_dropout_rate 0 \
--h3_rollout_field_floor 0.3
```

*Variations:* `--h3_guidance_scale_sigma_max 0.9` at the same cost fits the style data better, at the price of a
shorter field and half the conditional gain: part of the style appears on prompts without the trigger. Rollout 4
steps with window 2 (3x a plain step) changes no metric, and combined with the sigma cap it removes the
conditionality altogether.

### Full-parameter BF16 training

`minimax_h3_train.py` updates the entire transformer and writes a native MiniMax H3 BF16 checkpoint. It is separate from the
LoRA entry point in [Start LoRA training](#start-lora-training). Full training requires the ordinary BF16 FL2VA or Ref2VA checkpoint; FP8, ConvRot INT8,
LoRA initialization/merge weights, and architecture-changing options are rejected. `--h3_adaln_rank 16` is the one
architectural option it accepts: it trains the rank-reduced AdaLN projections (~77M parameters in place of 13.0B, in float32)
against a frozen timestep table and writes the checkpoint in the pruned layout, which loads like the released pruned
checkpoints without the flag and cannot be reduced a second time.

```shell
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch \
  --num_processes 1 --num_cpu_threads_per_process 1 \
  minimax_h3_train.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --h3_training_mode fl2va \
  --sdpa \
  --blocks_to_swap 8 \
  --adafactor_triton \
  --learning_rate 1e-6 \
  --max_train_steps 1000 \
  --save_every_n_steps 250 --save_state --autoresume \
  --output_dir output/full --output_name h3_full
```

The dense entry point defaults to BF16 weights, gradient checkpointing, manual-learning-rate Adafactor
(`scale_parameter=False relative_step=False warmup_init=False`), per-parameter optimizer steps during backward, stochastic
rounding back to BF16, `max_grad_norm=0`, and memory-efficient checkpoint writing. `--adafactor_triton` accelerates supported
contiguous BF16 matrices and falls back to the fused PyTorch Adafactor update for other tensors. It requires Triton and the
manual-learning-rate Adafactor arguments; if you replace `--optimizer_args`, include those three values.

| Option | Purpose |
| --- | --- |
| `--blocks_to_swap N` | Use backward-capable block swap for trainable weights. Increase `N` when the model weights do not fit; activations are a separate budget, served by gradient checkpointing and activation offload. Unlike LoRA block swap, do not add `--block_swap_h2d_only`. |
| `--block_swap_trainable_ring` | Use coalesced bidirectional block transfers and write updated weights back to pinned CPU masters. Requires block swap, gradient checkpointing, fused backward, and `--use_pinned_memory_for_block_swap`. |
| `--block_swap_ring_size N` | Number of reusable GPU block buffers for the trainable ring; `2` enables double buffering. |
| `--h3_adaln_rank 16` | Train the rank-reduced AdaLN projections instead of the full 13.0B AdaLN weights; the saved checkpoint carries `adaln_t_table` and the reduced weights (metadata `ss_h3_adaln_layout=pruned`). Rejected on already-pruned and INT8 ConvRot sources. |
| `--gradient_checkpointing_cpu_offload` | Offload checkpoint activations when long packed sequences still exceed VRAM. |
| `--mem_eff_save` | Stream native transformer tensors during `.safetensors` output; enabled by default. |
| `--no_mem_eff_save` | Write checkpoints with the ordinary safetensors writer instead of the streaming one; needs the whole checkpoint contiguous in host memory. |

The combination `--blocks_to_swap 48`, the trainable ring, and `--block_swap_ring_size 2` minimizes resident model blocks for
full-parameter training, but its host-memory and transfer requirements are high. Fit also depends on resolution, frame length,
and conditioning rows. `--gradient_checkpointing_cpu_offload` further reduces activation residency at the cost of additional
host memory and transfers.

For the trainable ring, add:

```shell
  --block_swap_trainable_ring \
  --use_pinned_memory_for_block_swap \
  --block_swap_ring_size 2
```

Pinned CPU allocation is substantial and depends on `--blocks_to_swap`; use ordinary backward-capable block swap when the host
cannot provide it. Dense training currently supports one process/GPU. For Ref2VA, use the Ref2VA BF16 checkpoint, a `ref2va`
conditioning cache plus reference latents, and `--h3_training_mode ref2va`.

### Saving and resuming

Add `--save_state` alongside `--save_every_n_steps N` or `--save_every_n_epochs N` to give every periodic checkpoint a resumable
`*-state` directory. Resume by pointing `--resume` at that directory, not at its neighboring `.safetensors` LoRA file:

```shell
--resume output/h3_style-step00001000-state
```

`--resume` restores the optimizer, scheduler, dataloader position, RNG streams, epoch, and displayed global step. The private draw streams of the sparse auxiliary objectives (rollout selection and stop sigma, sparse preservation, DOP, guidance probability) are reseeded from the restored global stream rather than restored in place, so a resumed run selects a different subset of steps than the uninterrupted one would have.
`--network_weights output/h3_style-step00001000.safetensors` only initializes LoRA weights for a new run; its optimizer,
scheduler, and step start at zero. Use `--save_state_on_train_end` if only the final state is needed. An incomplete state is
rejected instead of silently restarting at step zero. Add `--autoresume` to select the highest-step complete state matching
`--output_name` in `--output_dir`; an explicit `--resume` path takes priority.

Add `--async_checkpoint_save` to keep periodic saves off the training thread; see [Key options](#key-options).

For an external save trigger, provide one or both request paths:

```shell
--save_request_file /workspace/save-now.flag \
--save_and_stop_request_file /workspace/save-and-stop.flag
```

Creating the first file (for example, `touch /workspace/save-now.flag`) saves a step checkpoint after the current optimizer
step and continues training. Creating the second finishes the current optimizer step and uses the normal final save path before
exiting. Rank zero detects each request and broadcasts it to the other workers; the file is removed only after a successful save.
With `--save_state`, either request also writes resumable optimizer, scheduler, RNG, and dataloader state; otherwise it saves only
the model checkpoint. Requests received during gradient accumulation wait for the next completed optimizer step.

### Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `--sdpa`, `--flash_attn`, `--flash3` | required | Attention backend; one of the three must be passed. SDPA uses PyTorch without an additional attention package. Each FlashAttention flag needs its package, and `--flash3` needs a Hopper GPU. Both FlashAttention modes fall back to SDPA on padded batches. |
| `--h3_attn_auto_dispatch` | off | Prefer cuDNN SDPA for large maskless workloads. Requires `--sdpa`. Changes rounding; benchmark first. |
| `--h3_int8_attention {off,aux,train}` | `off` | Experimental native INT8-QK forward with BF16/FP16 P×V and an optimized training backward. `aux` affects only guidance and base-preservation teacher forwards; `train` also affects the trainable forward. Requires CUDA, Triton, and head width 128; masked or padded batches use the selected regular backend. Incompatible with `--compile`. |
| `--h3_block_sparse_kv_fraction F` | `0.0` | Experimental block-sparse attention over the packed sequence. Rows are grouped into 128-row blocks, and each query block attends to the top-scoring key blocks by the dot product of their means plus its own block. `0` disables it, `1.0` keeps every block and reproduces dense attention. Requires CUDA (`flex_attention`); the first steps pay a one-time compilation. Masked or padded batches fall back to the dense SDPA path. Selection is an approximation, so results differ from dense attention; validate before a long run. Incompatible with `--h3_int8_attention` and `--compile`. |
| `--h3_block_sparse_threshold F` | `0.0` | Alternative selection rule: keep the highest scoring key blocks until they hold this share of the score mass, instead of a fixed count. Takes precedence over `--h3_block_sparse_kv_fraction`; `1.0` keeps every block. |
| `--h3_block_sparse_start_block N` | `0` | Index of the first main block to run block-sparse; earlier blocks stay dense, keeping their full-sequence mixing exact. |
| `--h3_block_sparse_block_shape T,H,W` | off | Reorder target-video rows into 3D lattice tiles before block-sparse selection, then restore their original order. `T*H*W` must equal the 128-row block size; text, audio, and reference context remains dense. Requires a block-sparse selection option. |
| `--h3_lora_token_refiner` | off | Also place LoRA adapters on the two text token-refiner blocks. This experimental target adds eight adapter modules. |
| `--compile` | off | Regionally compile all H3 blocks with the selected backend/mode. Compatible with full or partial gradient checkpointing and with block swap; swapped Linear calls stay eager. |
| `--h3_fused_qk_norm_rope` | off | Use the custom Triton Q/K RMSNorm+RoPE kernel outside compiled graphs. It reduces separate kernel launches but changes BF16 rounding; benchmark it on the target workload. |
| `--h3_fused_indexed_adaln` | off | Fuse main-block RMSNorm with token-indexed AdaLN shift/scale for a frozen LoRA base. Requires CUDA and Triton; unsupported layouts, trainable norm/AdaLN parameters, and compiled blocks use the regular path. It changes BF16 rounding. |
| `--h3_fused_swiglu` | off | Fuse the SwiGLU activation in main and token-refiner feed-forward layers. Requires CUDA and Triton; unsupported layouts and compiled blocks use the regular path. It changes BF16 rounding, so benchmark and validate it before a long run. |
| `--h3_fused_elementwise` | off | Apply each main block's AdaLN modulation (`shift + normalized * (1 + scale)`), its two gated residual adds (`hidden + gate * branch`) and the LoRA delta add as single `addcmul` / `add(alpha=)` kernels. Eager mode only (compiled blocks fuse on their own). The product stays in the fp32 accumulator and is rounded once, so results can differ from the default operation order by BF16 rounding; validate before a long run. Precedence with `--h3_fused_indexed_adaln`: when that Triton kernel accepts the layout it handles the norm+modulation itself and the `addcmul` AdaLN path is bypassed for that call; the gated residual and LoRA fusions of this option remain active. |
| `--h3_compile_attention {inline,auto,opaque}` | `inline` | With `--compile`, how the attention kernel call is presented to Dynamo. `inline` traces it, keeping the partitioning every existing `--compile` run has. `opaque` wraps the kernel call in `torch.compiler.disable`, so each block compiles to exactly one graph before the kernel and one after it; use it when the compile log shows graph breaks inside the attention call (a FlashAttention binding without `torch.library` fake kernels), which otherwise cascade into small unfusable fragments around every block's kernel. `auto` chooses `opaque` for `--flash_attn`/`--flash3` and `inline` for `--sdpa`. Not combinable with `--compile_fullgraph`, which forbids any graph break; `auto` falls back to `inline` there. Numerics are unchanged: the wrapped call is the same eager kernel and autograd. |
| `--h3_swiglu_chunk_rows N` | `0` | Process each main-block feed-forward layer in sequence-row chunks to reduce peak VRAM. Start with `2048`; smaller values may save more memory but add overhead. Incompatible with `--compile`. |
| `--h3_gradient_checkpointing_cpu_offload_pin_memory` | off | Pin CPU-offloaded checkpoint activations for faster transfers. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload` and substantial free system RAM. |
| `--h3_reusable_activation_offload` | off | Reuse pinned CPU checkpoint buffers and prefetch activations in reverse block order. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload` and sufficient free system RAM. |
| `--gradient_checkpointing_cpu_offload_dtype` | `none` | Wire dtype of the offloaded activations. `fp8_e4m3` quantizes each large bf16/fp16 activation on the GPU (per-tensor scale) before the D2H copy, halving PCIe traffic and pinned host memory. Recomputation then consumes a lossy activation, so gradients differ from an exact run deterministically but not bit-identically. Payloads under 1 MiB, integer/boolean saves, and fp32 activations pass through unquantized. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload --h3_reusable_activation_offload`; keep `none` for exactness-sensitive runs. |
| `--h3_gradient_checkpointing_blocks N` | all 50 | Checkpoint only the last N main blocks. This explicit speed/VRAM trade-off requires `--gradient_checkpointing` and resident eager blocks. It can be combined with `--compile` (the eager blocks add one extra compiled variant) but not with block swap, whose streamed weights are only safe to reuse at recompute time. Each eager block retains its full activations, which at video sequence lengths costs several GB per block. Enabling `--h3_fused_qk_norm_rope`, `--h3_fused_indexed_adaln`, and `--h3_fused_swiglu` roughly halves each eager block's retained activation memory. |
| `--h3_shift_video` / `--h3_shift_audio` | `12.0` / `3.0` | Released per-modality flow shifts. Both derive from one shared coordinate; keep the defaults. |
| `--timestep_sampling` | `uniform` | Shape of the shared unshifted schedule. Keep `uniform` unless deliberately testing a different distribution. `sigmoid`, `logsnr`, and `sigma` are experimental alternatives; `shift` with H3's required `--discrete_flow_shift 1` is equivalent to sigmoid sampling. Model-specific dynamic-shift modes are rejected because H3 applies its own video/audio shifts afterward. |
| `--weighting_scheme` | `none` | Optional loss weighting. H3 bounds `sigma_sqrt` inverse-square weighting so that a near-zero draw cannot dominate a batch-1 optimizer step; `cosmap` is intrinsically bounded. |
| `--h3_sigma_sqrt_max_weight` | `10` | Maximum per-sample weight used by `sigma_sqrt`. It preserves extra emphasis at low sigma without the generic unbounded spike and has no effect with the default `none` scheme. |
| `--discrete_flow_shift` | `1.0` | Must stay at the default; H3 applies its own shifts. |
| `--h3_image_flow_shift` | auto | Fixed shift for image batches only. |
| `--h3_timestep_focus_probability` | `0` | Optional video/AV curriculum: draw this fraction of the uniform base schedule from `--h3_timestep_focus_min` through `--h3_timestep_focus_max` (defaults `0.4` and `0.8`) while retaining full-range samples. Image batches keep their resolution-aware schedule. Requires `uniform` and cannot be combined with min/max timestep clipping. |
| `--num_timestep_buckets` | off | Stratifies base timesteps across an epoch to reduce sampling imbalance. Incompatible with `--timestep_sampling sigma`. |
| `--h3_video_loss_weight` / `--h3_audio_loss_weight` | `1.0` | Modality weights. `--h3_loss_balance token` switches from equal modality means to element weighting. |
| `--async_checkpoint_save` | off | Hash and write periodic checkpoints on a background thread. The step pauses only for the CPU snapshot, so the `.safetensors` file lands shortly after the step instead of stalling training for the write. Saves stay ordered and never overlap; the final checkpoint and `--save_state` remain synchronous. |

### Memory and speed

The following options reduce different parts of model-weight residency. They are not all required, and some combinations are
mutually exclusive.

```shell
  --h3_convrot_int8 --h3_convrot_int8_fwd bf16 --h3_adaln_rank 16
```

| Option | Purpose |
| --- | --- |
| `--h3_convrot_int8` | Quantize the released BF16 checkpoint to ConvRot INT8 at load. Rejects `--fp8_base` and `--int8_convrot_base`. |
| `--h3_convrot_int8_fwd bf16` | Recommended. Evaluates the matmul in BF16 without changing stored weights. |
| `--h3_convrot_int8_bwd int8` | INT8 input-gradient path. Requires Triton; incompatible with `--h3_convrot_int8_fwd bf16`, which leaves no rotated activations. |
| `--h3_convrot_int8_lora_fused` | Fuse LoRA application into supported ConvRot INT8 linear calls. Requires an INT8 ConvRot base (online or pre-quantized), `--h3_convrot_int8_fwd int8`, `--h3_convrot_int8_bwd int8`, and triton. |
| `--fp8_base` | Scaled FP8. `--h3_fp8_quantization_mode` selects `block` (default, finest), `channel`, or `tensor`. |
| `--int8_convrot_base` | Load the released pre-quantized checkpoint instead of quantizing at load (see below). |
| `--h3_adaln_rank 16` | Reduce the AdaLN projections, the largest parameter group (13.0B of 33.1B), to ~77M. |
| `--h3_checkpoint_keep {none,attention,qkv}` | Trade activation memory for recompute under `--gradient_checkpointing`. `attention` keeps each block's fused attention output, so the backward recompute skips the attention forward; `qkv` also keeps the base QKV projection output (LoRA terms are still recomputed). One (`attention`) or four (`qkv`) rows-by-hidden BF16 tensors per checkpointed block; weights are never kept, so it composes with whole-block H2D-only block swap. Bit-identical to `none`. The saving exists only when attention runs as one fused kernel: SDPA's flash, cuDNN or efficient backend, or a flash-attn build that registers its `torch.library` ops. A batch that makes SDPA fall back to the math backend (unsupported mask, dtype or head width) is stopped with an error rather than silently recomputed. Rejected with `--gradient_checkpointing_cpu_offload`, `--compile`, `--h3_int8_attention train`, block-sparse attention, `--block_swap_granularity layer`, and (`qkv` only) `--h3_convrot_int8_lora_fused`. |

To train against the released pre-quantized transformer instead, pass it with `--int8_convrot_base`:

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --int8_convrot_base --sdpa --mixed_precision bf16 \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16
```

`--h3_convrot_int8_fwd {int8,bf16}`, `--h3_convrot_int8_bwd`, and the fused ConvRot LoRA path also apply to this
pre-quantized base, whose compact AdaLN projections are already baked in.

Generic LoRAs passed through `--base_weights` are merged into the frozen base at load, including a pre-quantized ConvRot base.

`--h3_overlay_weights` applies a LoRA as a separate frozen module instead of merging it: base weights are not modified, so
it also works on an INT8 ConvRot base. The overlay is excluded from the optimizer and from saved checkpoints, and is active
on all forwards, including the base-preservation reference. `--h3_overlay_weights_multiplier` (default `1.0`, negative
allowed) scales its delta. Incompatible with full fine-tuning.

`--h3_adaln_rank` avoids quantizing the AdaLN projections and composes with either quantization mode. It is rejected on
already-pruned and INT8 ConvRot checkpoints.

Block swapping streams frozen weights from host memory. It is valid only while the base is frozen:

```shell
  --blocks_to_swap 40 \
  --block_swap_h2d_only \
  --block_swap_ring_size 2
```

`--use_pinned_memory_for_block_swap` enables direct asynchronous host-to-device transfers at the cost of locked host RAM; time a few steps with and without it, and leave it disabled
when pinned allocations stall or fail. `--block_swap_granularity layer` streams individual `Linear` layers through the same
H2D-only ring and supports all 50 blocks, at the cost of more transfers; use the default `block` granularity when it fits. It
cannot be combined with `--h3_convrot_int8` or `--int8_convrot_base`. Add
`--gradient_checkpointing_cpu_offload` when sequence length would otherwise exceed VRAM, and set
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

For limited VRAM on a host with enough RAM for the BF16 checkpoint and swapped blocks, the loader can reduce and quantize
weights while placing swapped blocks on CPU:

```shell
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --h3_adaln_rank 16 \
  --h3_convrot_int8 --h3_convrot_int8_fwd bf16 \
  --blocks_to_swap 48 --block_swap_h2d_only --block_swap_ring_size 2 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 --save_state --autoresume \
  --output_dir output --output_name h3_style
```

If this configuration exceeds available VRAM, keep `--blocks_to_swap 48` — the maximum without layer granularity, which is
rejected here because it bypasses the ConvRot INT8 forward — and use `--block_swap_ring_size 1`, trading throughput for lower
device residency. Block-swap loading materializes CPU-master copies only for swapped blocks, so `--blocks_to_swap` affects both
host and device residency.

When pinned host memory is the binding constraint,
`--h3_reusable_activation_offload --gradient_checkpointing_cpu_offload_dtype fp8_e4m3` halves the offloaded buffer, since
an `fp8_e4m3` element is one byte against BF16's two.

DataLoader tuning affects cache delivery, not transformer compute. Increase `--max_data_loader_n_workers` only when storage or
CPU loading leaves the GPU idle; `--persistent_data_loader_workers` avoids restarting those workers each epoch, and
`--dataloader_prefetch_factor N` controls queued batches per worker. `--dataloader_pin_memory` enables pinned batch buffers and
non-blocking device copies, but consumes locked host RAM.

### Training modes

Extension, keyframes, and masking add or pin conditioning rows. Keyframes, masking, and `per_row_sigma` extension also combine
with `--h3_training_mode ref2va` / `ref2va_omni` and their reference caches; other target-derived conditioning needs a `t2va`
cache. Observed-modality training and the two jitters place no
conditioning rows and work with any cache. For observed-modality training,
`target_modalities = ["video", "audio"]` is required and each target video must contain its synchronized soundtrack — an audio-only dataset has no
video rows to observe, so `--h3_observed_modality` cannot be used with it. Conditioning an audio target on video is still
possible through Ref2VA references instead, which supply an arbitrary conditioning video rather than the target's own track; see
[Dataset](#dataset). The observed modality remains in the packed attention sequence but its
loss weight is forced to zero; this isolates direct supervision, not H3's shared attention parameters.

| Option | Trains |
| --- | --- |
| `--h3_observed_modality {video,audio,random}` | Video-to-audio, audio-to-video, or one adapter covering both plus joint |
| `--h3_extension_video_frames N` / `--h3_extension_audio_latents N` | Continuation from an observed prefix. Counts are in **latent** units and each must be shorter than its target; the two are independent, so setting one leaves the other generated in full. Under Ref2VA only the `per_row_sigma` route is supported |
| `--h3_extension_probability P` | Train the extension recipe on a synchronized random fraction of steps; the rest train the plain objective. Requires the extension flags. `1` (default) applies it every step |
| `--h3_keyframe_anchors first,11,last` / `--h3_keyframe_random_count N` | Interpolation from arbitrary target-frame guides. Supports FL2VA/`t2va` and Ref2VA/Ref2VA-Omni caches with video targets |
| `--h3_guide_specs "0:2:4;21:0:8"` | Ref2VA target-derived video/audio guides as `pixel_start:video_latents:audio_latents`. Separate guides with `;`; negative starts count from the end. Visual starts must lie on a cached video VAE-window boundary and audio starts must be multiples of 3 pixel frames. Guide rows are pinned and excluded from loss |
| `--h3_mask_mode {off,box,border,segment,dataset}` | Inpainting, outpainting, temporal infilling from a procedural mask, or the region the dataset authored. `off` is the default. Also available under Ref2VA |
| `--h3_mask_probability P` | Train the masked recipe on a synchronized random fraction of steps; the rest train the plain objective. Requires `--h3_mask_mode` or `--h3_mask_audio`. `1` (default) applies it every step |
| `--h3_frame_sigma_jitter 0.2` | Spreads target-frame noise levels across the schedule in one step. Supported by native T2VA/I2VA/FL2VA/L2VA/Ref2VA caches, including guidance-consistent loss, and skipped for images; cannot be combined with in-target observed-row options or sigma-dependent loss weighting; `0` disables it |
| `--h3_spatial_density_jitter 0.2` | Perturbs the area normalization of the spatial RoPE grids each step, drawn log-uniformly from `[1/1.2, 1.2]`, so fixed-resolution data still trains a range of token spacings. One factor covers every grid in the packed sequence; `0` disables it |
| `--h3_caption_dropout_rate 0.1` | Trains the unconditional branch; requires `--cache_guidance_empty` |

Extension, keyframes, and masking all claim the observed rows, so **only one may be active on a step**.

**Mixing recipes.** `--h3_mask_probability` and `--h3_extension_probability` turn the choice into one categorical draw per
optimizer step. The draw is shared by every item of the batch, by the guidance and preservation branches, and across distributed
ranks. Setting both allows masking and extension in the same run, provided the two probabilities sum to at most `1`; the draw
selects at most one of them per step and the remainder trains the plain objective. No loss is rescaled by its probability.
`h3/recipe_mask_active` and `h3/recipe_extension_active` are logged only once a probability below `1` is set.

**Observed modality.** The observed side is pinned to the noise level the release already uses for conditioning of that kind.
Datasets must cache both modalities. `random` redraws the task each step; validation then reports `val/joint`, `val/v2a`, and
`val/a2v` separately.

**Extension.** `--h3_extension_route` chooses the presentation: `condition_rows` (default) duplicates the context as clean rows,
matching the released keyframe contract but costing sequence length; `per_row_sigma` pins it in place with no extra tokens, at the
cost of intra-block noise levels the released weights have not seen. The observed span is removed from the loss. Under
`--h3_training_mode ref2va` / `ref2va_omni`, `condition_rows` is rejected: duplicating the context needs packer support the
Ref2VA layout does not have, so pass `--h3_extension_route per_row_sigma`.

**Keyframes.** Entries are `first`, `last`, or a latent frame index; negative indices count backward from the final latent
window. Anchors stay in the loss, matching the released contract.
`last` is the final *pixel* frame, not the same anchor as the integer `frames - 1`. A `t2va` cache is text-only, so the
conditioner never sees the frames those anchors pin; cache with `--h3_keyframe_visuals first,11,last` (EXPERIMENTAL, `--task
t2va` only, entries are decoded *target-video* frames) to present them to Qwen3-VL as picture spans ahead of any Qwen control
spans, restoring the visibility the released `i2va`/`fl2va`/`l2va` presentations have. The list is part of the text-cache
identity, so changing it rebuilds the cache under `--skip_existing`; latent caches and DiT rows are untouched, and a run whose
anchors name other frames than the cache presents is only warned about, once.
Under Ref2VA/Ref2VA-Omni, target-derived guide rows follow the ordinary reference rows and share the target timeline, while
Qwen3-VL continues to see the cached ordinary references. This combines global references with timeline anchors without
re-caching. It requires a video target; audio-only targets cannot supply guide frames.

**Video/audio guides.** `--h3_guide_specs` extends the same Ref2VA layout to a short video span, an audio span, or both at one
pixel-frame origin. In `0:2:4`, `0` is the decoded target frame, `2` is the number of clean target video latents, and `4` is the
number of clean target audio latents. Use zero for an absent stream. Multiple `;`-separated guides are packed in order after
ordinary references; visual rows use conditioning sigma `0.999`, audio rows use `1.0`, and only the original target rows are
returned to the loss. A negative origin counts backward from the decoded target end. Since training reads pre-cached target
latents rather than decoding and re-encoding a new subclip, a visual origin must be one of H3's VAE-window boundaries
(`0,1,5,9,...`); an audio span must start on its cached 40-Hz boundary, which occurs every three 24-fps pixel frames. The total span must fit the target. This option requires AV caches
when its audio length is non-zero and currently applies to `ref2va` / `ref2va_omni`, not FL2VA.

`--h3_guide_specs` extracts sparse video and/or audio spans from the cached target. `aligned_guide_indices` instead reads an
external visual-only video, normalizes it to the complete target geometry, and assigns it the complete target position grid.
Aligned external guides currently begin at target frame zero and cannot carry audio.

**Masking.** Masks are drawn per step, so the occlusion distribution changes without re-caching. Masks combine with `t2va` and
`ref2va`/`ref2va_omni` caches; `i2va`/`fl2va`/`l2va` caches reject observed rows. Under Ref2VA the observed rows are pinned
inside the target block, the reference rows keep their own conditioning noise level, and neither is scored.
`--h3_mask_min_fraction` and `--h3_mask_max_fraction` bound the generated fraction of **each masked axis**, defaulting to `0.25`
and `0.75`; `--h3_mask_audio` also hides a run of audio latents. Masks are reduced to the
`(1, 2, 2)` patch grid and a patch counts as generated when any latent inside it is, so at small latent resolutions a wide
fraction range can leave nothing observed.

A conditioning mask is not a loss mask: it says which pixels the model *observes* as clean context, while `loss_mask_directory`
and friends say which pixels are *scored*. Both may be set; the effective loss is their intersection.

**Dataset masks.** `--h3_mask_mode dataset` reads the observed region from the dataset instead of drawing one, for segmentation
maps, mattes, or hand-drawn regions.

| Field | Behavior |
| --- | --- |
| `conditioning_mask_directory` | Directory of mask images matched to targets by basename, exactly one per target |
| `conditioning_mask_path` | Per-record alternative in an image or video JSONL, resolved against the JSONL's directory; setting it together with the directory is an error |

Masks are still images (`png`, `jpg`, …), read as luma: pixels above `127` are **observed context**, everything else is the
region to generate. The same plane applies to every frame; video masks are not supported. The mask is resized to the
item's bucket resolution with nearest-neighbour sampling and no antialiasing (aspect ratio is not preserved, since a mask is
authored against its target), then reduced to latent cells: a cell is observed only when every pixel inside it is, which keeps
the region boundary on the generated side just as the patch reduction does. Masks are applied to the packed rows at train time,
so adding, editing, or removing one needs no re-cache.

Downstream the behavior is identical to a procedural mask, with four differences. `--h3_mask_min_fraction` /
`--h3_mask_max_fraction` do not apply. A mask that reduces to all-observed or all-generated is warned about once, naming the
item. `--h3_mask_audio` stays procedural: dataset masks are video-only, and declaring one on an audio dataset is an error. The
observed rows are shared by the whole batch, so a batch whose items carry different masks is rejected; use `batch_size = 1`. A
config that declares masks under a procedural mode, or selects `dataset` without them, is rejected when the dataset config is
parsed.

```toml
[[datasets]]
target_video_directory = "/data/clips"
target_modalities = ["video", "audio"]
conditioning_mask_directory = "/data/masks"   # clip_01.mp4 -> clip_01.png, white = observed
batch_size = 1
```

### Auxiliary objectives

Guidance distillation and active base preservation add no-gradient transformer forwards over the packed sequence, so their cost
grows with sequence length and is larger for reference-conditioned batches. Rollout supervision adds both: no-gradient forwards
to reach the state, and one gradient-carrying forward per supervised state. CREPA reuses features from the main forward and does
not add a complete H3 forward.

Every objective here is off unless its flag is given, and a run without them trains exactly as it did before they existed.

| Option | Purpose |
| --- | --- |
| `--h3_guidance_distillation_scale S` | Guidance-consistent objective using cached empty-text conditioning at scale `S`. `--h3_guidance_loss_form` selects `normalized` or `contrastive`; for the same predictions and scale, the contrastive form is `S²` larger. |
| `--h3_guidance_scale_range 2.5,3.5` | Draw the distillation scale uniformly in `[LOWER, UPPER]`, once per micro-batch sample and step, instead of pinning one value; the adapter then learns a family of guidance strengths rather than a single point. Replaces `--h3_guidance_distillation_scale` and is rejected alongside it. `LOWER` must exceed `1`. The draw uses its own distributed-synchronized generator, so adding it leaves every other random branch of a seeded run untouched; validation reads the midpoint so its loss stays comparable across evaluations. Composes with the sparse probability and with both loss forms and schedules. |
| `--h3_guidance_distillation_probability 0.5` | Evaluate the empty-conditioning branch on a synchronized random fraction of batches, skipping its extra forward on the rest, and scale the guidance correction by `1 / probability`. `1` (default) applies the objective every batch; smaller values preserve the expected loss, but rare larger corrections are not optimizer-equivalent to applying the dense objective every step. |
| `--h3_guidance_loss_schedule {sigma,constant}` | `sigma` (default) scales guidance from `1` at the clean endpoint to the configured value at maximum noise, independently for video and audio. `constant` retains the configured scale everywhere. |
| `--h3_guidance_null_source {live,frozen}` | `live` (default) evaluates the null-conditioning branch with the trainable adapter active, so that branch drifts along with training. `frozen` disables the adapter for that forward only, giving the guidance correction a fixed base-model anchor. Requires a network that supports runtime disabling; it replaces the existing empty forward rather than adding one. |
| `--h3_fuse_frozen_teachers` | Experimental opt-in for runs that use both `--h3_guidance_null_source frozen` and base preservation. It batches the active empty-guidance and preservation teachers into one forward when their cached text layouts match, otherwise it safely uses the sequential path. It may increase transient teacher memory. |
| `--h3_guidance_cfg_zero` | CFG-Zero* rescale of the null branch before the guidance form is applied: per sample and per modality, `alpha = <conditional, null> / (‖null‖² + 1e-8)` projects the null field onto the conditional one, so a null branch orthogonal to the conditional field collapses instead of being extrapolated away from. |
| `--h3_base_preservation_loss_weight 0.02` | Recommended starting value. Penalizes drift from the frozen base's prediction and anchors to whichever base is loaded, quantized or not. |
| `--h3_base_preservation_probability 1.0` | Evaluate preservation on a synchronized random fraction of batches and scale active losses by `1 / probability`. `1` applies the objective every batch; `0.25` evaluates it on a quarter of them at four times the weight. |
| `--h3_guidance_null_anchor_weight 0.0` | Experimental. Penalizes movement of the EMPTY-prompt prediction away from the frozen base's, weighted by this value, and leaves the prompted prediction free; unlike `--h3_base_preservation_loss_weight` it does not pull the prompted branch and so does not fight the data term directly. Requires `--cache_guidance_empty`; costs one gradient-carrying and one no-gradient forward per prompted step and is skipped on caption-dropout steps. Its term is logged as `loss/guidance_null_anchor` and left out of `loss/average`. Judge it on `val/drift/prompted_rel`, not on `val/field`, which is built from the difference this term holds still. `0` disables. |
| `--h3_guidance_null_anchor_probability 1.0` | Evaluate the null anchor on a synchronized random fraction of prompted steps and divide the active term by it, so the expected gradient is the dense anchor's at a fraction of its two forwards a step. Reported as `h3/null_anchor_active`. |
| `--h3_rollout_supervision` | Supervise the adapter at states reached by its own sampler, against a privileged frozen teacher. Requires `--h3_rollout_teacher_config`. See [Rollout supervision](#rollout-supervision). |
| `--h3_rollout_teacher_config teacher.toml` | Dataset config for the teacher: the same clips and captions as the training set, with extra information about each clip in its cache. Its privilege channel is validated before the model loads. |
| `--h3_rollout_steps 2` | No-gradient Euler steps taken from pure noise to reach the supervised state. Default `2`. |
| `--h3_rollout_window 1` | Supervised states taken from the stop sigma onward, on the same step size. Each one adds a gradient-bearing student forward and a no-gradient teacher forward, so this is the parameter that governs cost. Default `1`, maximum `3`. Above `1`, a window whose stride exceeds the drawn stop sigma puts several sub-steps on the same floor value, which supervises one state repeatedly. |
| `--h3_rollout_probability 0.5` | Fraction of steps that take the rollout path; the rest train normally. `1` applies it every step. Default `0.5`. |
| `--h3_rollout_teacher_privilege {auto,qwen,reference,keyframe,caption}` | `auto` (default) reads the caches and resolves the channel. Every channel is a difference between the teacher's cache and the student's for the same item: a teacher whose Qwen assets or endpoint task the student also carries is not privileged. Only the resolved channel's presentation reaches the teacher, so pin one to measure a particular channel on a corpus that carries several. |
| `--h3_rollout_stop_shifted` | Draw the stop sigma uniformly on the shifted video schedule instead of the shared unshifted one. Video and audio stay synchronized either way. |
| `--h3_rollout_fused_teacher` | Compute the student and teacher predictions for one supervised state in a single pass over the blocks, so each swapped block is streamed once per state instead of once per arm. May increase transient memory. Rejected together with `--h3_int8_attention aux` and with block-sparse attention (`--h3_block_sparse_kv_fraction` or `--h3_block_sparse_threshold` above `0`), whose per-sequence attention plan the fused pass cannot keep separate per arm. |
| `--h3_rollout_field_floor 0.0` | Experimental. Length floor on a frozen-null proxy of the guidance field at the supervised rollout states: `W * relu(1 - ‖g' - e‖ / ‖g - e‖)^2` with the frozen base's prompted `g` and empty `e` and the student's prompted `g'` at the same state. In the default `self` mode holds `‖g' - e‖` at no less than the checkpoint's field length and leaves the direction to the teacher term (see `--h3_rollout_field_floor_direction`); it does not bound the gap to the student's own empty branch, so pair it with `--h3_guidance_null_anchor_weight`. Requires `--h3_rollout_supervision` and `--cache_guidance_empty`; two extra no-gradient frozen forwards per supervised state. `0` disables. See [Field length floor](#field-length-floor). |
| `--h3_rollout_teacher_magnitude_weight 1.0` | Experimental. Below `1` the rollout teacher term is split in field space against the frozen empty branch: direction (the student's field rescaled to the teacher's field length, matched to the teacher's field; no gradient on the length) plus `M *` a separate magnitude term. `0` takes the teacher's direction only and leaves the field's length to the floor, removing the measured conflict between the two. Needs `--h3_rollout_field_floor` above `0`. See [Field length floor](#field-length-floor). |
| `--h3_rollout_field_floor_direction {self,teacher}` | Which direction the floor measures the student's field along. `self` (default) floors the plain length, whose gradient lengthens the field along the student's own current direction. `teacher` floors the projection of the student's field onto the privileged teacher's field at the same state, so lengthening in a wrong direction earns nothing and the gradient turns the field toward the teacher as it lengthens it. No extra forward. Needs the floor above `0`. |
| `--h3_rollout_field_floor_sigma_max 1.0` | Apply the floor only at supervised states whose shifted video sigma is at most this value; states above it drop out of the floor's mean. Needs the floor above `0`. |
| `--h3_rollout_field_cap 0.0` | Upper bound on the field-length ratio the floor scores: adds `relu(ratio - cap)^2` under the floor's weight, holding the field in a band `[1, cap]`. An over-long field shows as burnt colour and contrast, and once co-adapted into the weights it is not undone by an inference multiplier. Under `--h3_rollout_field_floor_direction teacher` the ratio is the signed projection onto the teacher's direction and the cap bounds that projection, not the field's total length. Needs the floor above `0` and a value above `1`. |
| `--h3_rollout_null_anchor_weight 0.0` | The null anchor at the supervised rollout states: the student's empty-prompt prediction there (adapter on, gradient) is held to the frozen base's, weighted by this value. One graded forward per supervised state on rollout steps only, against the frozen empty arm the floor already evaluates; the data-step anchor can then be dropped to move the anchor onto the states generation walks. Requires `--cache_guidance_empty`; rejected with `--h3_adapter_prompt_only`. |
| `--h3_rollout_prefix {student,teacher}` | Who walks the no-gradient prefix of the rollout. `student` (default) is on-policy. `teacher` walks it with the frozen privileged teacher and the student's window refines the states it reaches -- a hybrid policy that keeps the teacher off states the student built from a conditioning it does not share. Same forward count. |
| `--h3_rollout_stop_min 0.0` | Lower bound of the rollout stop draw, on the coordinate the draw is made on (unshifted base, or the shifted video sigma under `--h3_rollout_stop_shifted`). Without the shifted stop, `--h3_timestep_focus_max` at or below it gives the data term and the rollout term disjoint base-sigma bands. `0` draws the whole range. |
| `--h3_adapter_prompt_only` | Run every empty-prompt forward with the adapter off, so the student's empty branch is the frozen checkpoint's by construction. A guidance-distilled checkpoint never evaluates its empty branch at inference. Makes the live-null degeneracy impossible and `--h3_guidance_null_anchor_weight` redundant (the two are rejected together, as is `--h3_caption_dropout_rate` above `0`). See [Stability](#stability-controls). |
| `--h3_measured_variance_weighting curve.json` | Weight each sample's loss by the inverse of the target dispersion proxy measured at its shifted sigma, read off a `probe_g_curve` report (`raw^2 - g^2` per bucket and modality; a proxy, not a variance), normalised to mean 1 and capped by `--h3_measured_variance_weight_max` (default `4.0`). A heuristic sigma reweighting that changes the objective; composes multiplicatively with `--weighting_scheme`. See [Stability](#stability-controls). |
| `--h3_guidance_scale_sigma_max 1.0` | Cap on the guidance target's schedule: above this shifted sigma the distillation scale is `1` and the step trains on the plain data target. The checkpoints' implied scale runs to 11-16 above sigma 0.9, where the data are nearly noise. `1.0` applies the schedule everywhere. |
| `--h3_guidance_audio_scale` | Distillation scale for the audio half of the guidance target; unset uses the configured video scale (including a per-sample `--h3_guidance_scale_range` draw) for both. At exactly `1` the audio target is the plain data target bit for bit. The released checkpoints' audio field is not a clean amplification (its implied scale is not constant across sigma and the residual is nearly all systematic at the top), so `1.0` keeps the audio target plain while the video keeps the guidance form. At least `1`. |
| `--h3_validation_multipliers 0.75,1.25` | Repeat validation's data-state pass at these adapter multipliers and report `val/m<multiplier>/velocity_err_rel`, `field`, `field_cos`, `drift/prompted_rel`: how the adapter responds to the strength knob users turn at inference. One extra pass per multiplier; the rollout probe is not repeated. |
| `--h3_term_grad_every 0` | Every N steps differentiate each loss term (main objective, null anchor, field floor, rollout anchor, base preservation) on the adapter separately and report `h3/grad/<term>_norm` and pairwise `h3/grad/cos_<a>_<b>`: whether the terms agree or fight. One extra backward per term on those steps and one flattened fp32 gradient per term held at once. Single-process only and only with `--blocks_to_swap 0`: a block-swap offloader retires blocks on backward-hook firings, which a separate per-term backward would trip; skipped with a warning otherwise. |
| `--h3_adapter_stats_every 0` | Every N steps report `h3/adapter/delta_norm` (Frobenius norm of the weight delta over the unsplit Linear LoRA modules; split and convolutional adapters are not counted) and, with an EMA, `h3/adapter/ema_rel_dist` (relative distance of the live parameters from their EMA). |
| `--h3_train_sigma_bins` | Report the data step's video loss under `loss/video/bin<k>` for its shifted-sigma bin (edges 0.5, 0.8, 0.9). |
| `--h3_validation_std` | Report `<key>_std`, the per-clip sample standard deviation, beside each pooled validation metric that at least two clips contributed to. |
| `--h3_adapter_ema_decay 0.0` | Keep an exponential moving average of the adapter's trainable parameters, updated after every optimizer step, and save it beside each scheduled checkpoint as `<output_name>-ema-step<N>.safetensors` (an ordinary adapter file). Damps the step-to-step swing between competing terms without changing the objective. `--h3_validate_ema` validates the average instead of the live adapter. |
| `--h3_dop_loss_weight 0.0` | Differential Output Preservation, off at the default: penalizes LoRA drift from the frozen base under a trigger-free rewrite of each caption. |
| `--h3_dop_probability 1.0` | Evaluate DOP on a synchronized random fraction of prompt-conditioned batches and inverse-probability scale its loss. |
| `--crepa` | Temporal representation alignment for video training. |

Treat `--h3_base_preservation_loss_weight 0.02` as an initial value rather than a universal setting. Its effect depends on
training length, quantization, adapter rank, dataset, and learning rate. Higher values such as `0.05` can preserve the base very
strongly but substantially slow concept learning. Monitor training and validation samples, and reduce or disable the objective
when preservation dominates. The value can be changed when resuming training.

#### Differential Output Preservation

DOP is useful when a LoRA should specialize a trigger while retaining the base model's response to the surrounding class. For
a caption such as `sks woman walking in a park`, caching with `--h3_dop_trigger sks --h3_dop_class_prompt woman` creates the
alternate presentation `woman walking in a park`. Training then compares LoRA-on and LoRA-off predictions under that alternate
presentation using the same noisy video/audio inputs, timesteps, generated-region masks, and modality weights. This differs from
base preservation, which compares the models under the original trigger-bearing caption.

Pass the trigger and class phrase to both text caching and training:

```shell
python minimax_h3_cache_text_encoder_outputs.py ... \
  --h3_dop_trigger sks --h3_dop_class_prompt woman

accelerate launch minimax_h3_train_network.py ... \
  --h3_dop_trigger sks --h3_dop_class_prompt woman \
  --h3_dop_loss_weight 0.02 --h3_dop_probability 0.25
```

The trigger must occur as a standalone term in every cached caption. The cache records an identity for the exact trigger/class
pair and training rejects stale or mismatched caches. DOP is currently supported for the T2VA/FL2VA family, not Ref2VA. It adds
one frozen and one trainable transformer pass on active steps; the trainable DOP pass is backpropagated before the ordinary pass
so variable prompt lengths remain compatible with activation checkpointing and block swapping.

CREPA aligns projected features from an earlier block with a later block (`mode=backbone`) or with frozen DINOv2 features
(`mode=dino`). Only generated video rows participate; image, audio-only, and video-observed batches are skipped.

```shell
accelerate launch minimax_h3_train_network.py ... \
  --crepa mode=backbone student_block=16 teacher_block=33 weight=0.05 tau=1 neighbors=2
```

| Field | Default | Meaning |
| --- | ---: | --- |
| `mode` | `backbone` | `backbone` uses a later H3 block as teacher; `dino` uses cached DINOv2 features. |
| `student_block` / `teacher_block` | `16` / `33` | Zero-based blocks; in `backbone` mode the student must precede the teacher, and both must be below the transformer's block count, checked when CREPA is installed. |
| `dino_model` | `dinov2_vitb14` | `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, or `dinov2_vitg14`. Must match the cache. |
| `weight` | `0.05` | Multiplier on the alignment objective. |
| `tau` | `1.0` | Temporal distance temperature; offset `d` is weighted `exp(-|d| / tau)`. |
| `neighbors` | `2` | Frames compared on each side, in addition to the same-frame target. |
| `schedule` / `warmup_steps` / `max_steps` | `constant` / `0` / `0` | Weight schedule; `max_steps` required for `linear` and `cosine`. |
| `normalize` | `true` | L2-normalize before alignment. |
| `cutoff_step` | `0` | Disable at this step; zero leaves it enabled. |
| `similarity_threshold` / `similarity_ema_decay` / `threshold_mode` | unset / `0.99` / `permanent` | Disable once EMA alignment reaches the threshold. |

For a DINOv2 teacher, cache its features after the latent cache:

```shell
python minimax_h3_cache_dino_features.py \
  --dataset_config dataset.toml --dino_model dinov2_vitb14 \
  --device cuda --atomic_cache_writes --skip_existing
```

DINOv2 is fetched through Torch Hub by default; use `--dino_repo` for a local checkout and `--torch_hub_dir` for a persistent
model cache. `validation_datasets` entries are cached in the same invocation, and the cached `dino_model` must match the one
training requests. Frames follow the upstream DINOv2 transform — bicubic short-edge resize to 518 with antialiasing, then a
center crop — so rectangular bucket frames keep their aspect ratio. Each cache records the preprocessing version it was
written with; `--skip_existing` rebuilds any cache written by an older pipeline instead of mixing conventions.

CREPA's projection head is training state, not part of the inference LoRA; it is saved separately as `h3_crepa.safetensors`.

### Diagnostics

Three diagnostics exist for investigating a run and are **not** part of any recipe: all are documented in `--help` as debug
only and subject to removal, and none should appear in a command line you intend to keep.

| Option | Purpose |
| --- | --- |
| `--h3_profile_steps N` | After three warm-up optimizer steps, record the next `N` with `torch.profiler` and print one table of device time per kernel category — attention, GEMM, host↔device swap (`Memcpy HtoD/DtoH`, i.e. block swap and offload traffic), elementwise/other — plus the idle time of the window outside the union of all kernel intervals. Kernels and copies overlap across streams, so the category shares can add up to more than the busy time. The forward carries `h3.adaln`, `h3.attn`, `h3.mlp` and `h3.lora` labels, visible when the trace is inspected; outside profiling they cost a flag read per block, negligible next to its compute. Written to the log and to `<output_dir>/<output_name>_profile.txt`, which also lists the 30 heaviest kernels by name with the category each was filed under, so a kernel the patterns miss can be fixed from the file without a rerun; the profiler then stops and training continues. The run must be at least `3 + N` optimizer steps long, and a run that ends earlier still writes what the window holds. `0` (default) disables. |
| `--h3_validation_field_probe` | Report what the adapter did to the guidance field, measured against the frozen base on the validation items. Requires a validation set and `--cache_guidance_empty`, and is incompatible with `--base_weights`, which merges an adapter into the checkpoint it measures against. `val/velocity_err` is an energy ratio, so interpolate on its square root rather than on it. On a full fine-tune the reference is taken at step 0, when the weights still are the checkpoint, and written to `<output_name>_field_probe_base.pt` beside the checkpoints: start such a run with `--validate_at_start`. A resumed run reads the file back and refuses it when the checkpoint, validation config, seed or item cap differ from the ones it was taken under; single process only. `val/drift/prompted_rel` reports how far the PROMPTED prediction moved from the frozen base's, relative to the base field, and is the one number here that the empty-branch anchor cannot hold still by construction. |
| `--h3_validation_rollout_probe N` | Roll the adapter and the frozen base from the same noise for `N` Euler steps and report, at the states reached, the adapter's clean-clip error and the surviving field. The field probe measures on noised data states, which generation never visits. Costs `2N+4` no-grad forwards per validation. `0` (default) disables. Incompatible with `--base_weights` for the same reason as the field probe. Not available on a full fine-tune, which keeps no frozen base to evaluate at the walked-to state. |

### Validation

```shell
accelerate launch minimax_h3_train_network.py ... \
  --validation_dataset_config validation.toml \
  --validate_at_start --validate_every_n_steps 100 \
  --validation_timestep_bins 4 \
  --validation_min_timestep 100 --validation_max_timestep 900 \
  --max_validation_items 8
```

The validation TOML uses the same format and prebuilt caches as training, with `batch_size = 1`. Evaluation is deterministic:
each index and bin draws stable noise from `--validation_seed`, which defaults to the training seed. RNG state is restored
afterwards, so enabling validation does not change training randomness. Validation measures one fixed recipe — the masked one in
a run that mixes masking and extension — so successive numbers stay comparable.

Bins are equal-width midpoints between `--validation_min_timestep` and `--validation_max_timestep` in the shared *unshifted*
coordinate; video and audio then receive their own shifts, and image batches use the same `--h3_image_flow_shift` or automatic
image shift as image training. TensorBoard receives `val/loss`, `val/loss/video`, `val/loss/audio`, and `val/loss/bin_NN`. CREPA
and base-preservation are excluded. Guidance distillation is applied during validation whenever
`--h3_guidance_distillation_scale` is set, so the cached empty-text conditioning must be present. With `--weighting_scheme`
active, sample weights fold into both the numerator and the denominator; `val/loss` pools those sums across bins with the
modality balance weights, while `val/loss/video` and `val/loss/audio` are plain means that ignore them.

When generated or reference modalities are randomized during training, validation evaluates every enabled task deterministically
and prefixes metrics with `joint`, `v2a`, `a2v`, and, when applicable, `ref_av`, `ref_video`, or `ref_audio`. This adds validation
forwards only for the extra tasks.

### Sampling during FL2VA training

Training-time sampling is accepted only for `--h3_training_mode fl2va`; every other mode rejects `--sample_prompts` at
startup and is sampled with `minimax_h3_generate_video.py` instead.


```shell
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_at_first --sample_every_n_steps 50
```

Prompt files use the shared Musubi syntax, with `--i` and `--ei` for first and last keyframes:

```text
A traveler walks through a sunlit valley while birds sing. --w 672 --h 384 --f 124 --d 42 --s 20
A slow camera move across a coastal landscape. --i start.png --ei end.png --w 672 --h 384 --f 124 --d 42 --s 20
```

Samples are written to `OUTPUT_DIR/sample` as synchronized MP4 files with JSON sidecars, using the LoRA currently attached.

## Inference

```shell
python minimax_h3_generate_video.py \
  --model /path/to/minimax_h3_fl2va_bf16.safetensors \
  --text_encoder /path/to/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --prompt "A traveler walks through a sunlit mountain valley while birds sing." \
  --duration 5 --ratio 16:9 --steps 20 --seed 42 \
  --output output.mp4
```

| Option | Purpose |
| --- | --- |
| `--first_frame` / `--last_frame` | Keyframe conditioning at the ends. |
| `--keyframe INDEX:PATH` | Image guide at an arbitrary latent frame, repeatable; negative indices count from the end. With the Ref2VA checkpoint it may be combined with ordinary image/video/audio references. |
| `--guide_image PIXEL_FRAME:PATH` / `--guide_video PIXEL_FRAME:PATH` / `--guide_audio PIXEL_FRAME:PATH` | Comfy-style Ref2VA guides on the decoded 24-fps pixel timeline. Repeat them; image/video and audio entries with the same origin form one AV guide. Negative origins count from the end. Videos are cropped to the remaining target and to a valid `17k+5` length (batches shorter than five frames use their first image); audio is cropped to the remaining audio timeline. Ordinary references may be present or omitted. |
| `--reference_image` / `--reference_video` / `--reference_audio` | Ref2VA references; audio must accompany an image or video. A reference video's own soundtrack is included automatically, so do not also pass it as `--reference_audio`. Requires the Ref2VA checkpoint. |
| `--reference_image_size_mode` | Ref2VA image sizing: `short_edge` keeps the released behavior; `target_area` preserves aspect ratio and uses approximately the target output area. |
| `--reference_image_short_edge` | Reference-image short edge in `short_edge` mode (default 2048). |
| `--reference_image_max_pixels` | Optional pixel-area cap in `target_area` mode; `0` uses the target output area. |
| `--reference_video_short_edge` | Reference-video short edge (default 768, minimum 32). Lower values reduce Ref2VA reference rows, speed cost, and VRAM. |
| `--reference_video_max_pixels` | Maximum pixels per reference-video frame after aspect-preserving resize (default 768×1344, minimum 1024). The cap is enforced on the final 32-aligned dimensions, so extreme aspect ratios are downscaled rather than rounded back over it. |
| `--reference_video_fps` | Caching and training only; no inference flag. Subsample every reference video to this many frames per **source** second so the whole clip conditions the model instead of only its opening span. `0` (default) truncates the reference to the target's frame count. Frame `k` is source frame `round(k × source_fps / F)`, and the result is snapped to the nearest legal reference length (`1` or `17n+5`) that still fits the target's frame budget, truncating or padding with the final frame to reach it. The subsampled reference video spans the whole clip while its paired soundtrack still covers only the clip's opening span. |
| `--lora_weight` / `--lora_multiplier` | Attach saved adapters. ComfyUI key names are converted; incompatible module keys are rejected. |
| `--h3_learned_context` | Prepend a ComfyUI-compatible H3 learned context to the Qwen prompt output. Repeat the option to concatenate contexts in command-line order. It composes with all attached LoRAs. |
| `--h3_learned_context_multiplier` | Scale the corresponding learned context; omitted entries default to `1`. A value of `0` removes that context completely. Negative values are allowed for experimentation but are not a mathematically guaranteed inverse because context tokens influence nonlinear attention rather than adding a direction directly to the latent. |
| `--steps` | Sigma grid points including terminal zero, so `20` runs 19 evaluations. |
| `--latent_upscaler` / `--latent_upscale_scale` | Apply a trained latent upscaler before decoding. The default scale `1.0` leaves the latent unchanged. |
| `--first_pass_scale` / `--first_pass_steps` | Denoise first at this canvas scale, then refine at full size. `0` disables the first pass; its step count otherwise defaults to `--steps`. |
| `--second_pass_strength` | Fraction of the denoising schedule used by the full-size refinement; default `0.5`. |
| `--first_pass_lora` | Apply attached LoRAs to the first pass as well as the refinement; off by default. |
| `--h3_null_guidance_scale 0.0` | Classifier-free guidance at sampling time against the base checkpoint's empty-prompt branch: the prompted forward runs with the adapters, the null-prompt forward with them off, combined as `null + scale * (prompted - null)`. Restores the amplification a plainly trained adapter lost (its field settles near `1/w` of the base's, so a scale near `w`, measured at 4-5, brings it back). Two forwards per step. Needs `--lora_weight`; skipped on a first pass that runs without the adapters. |

The `--reference_video_*` values enter the cache identity: pass the same value to `minimax_h3_cache_latents`,
`minimax_h3_cache_text_encoder_outputs` and training, and re-cache both stages when one changes. The sizing values
(`--reference_image_short_edge` and the max-pixel caps) must match at inference too; `--reference_video_fps` has no
inference flag, because a reference supplied at generation time is sampled by the caller.

For a conditioned still image, select `--h3_image_mode first` with `--first_frame`, or `first_last` with both endpoint
images, and use an image extension for `--output`. The default 5-frame grid is decoded through the video VAE and
`--h3_select_frame` chooses the saved frame; the audio VAE is not required for image-file output.

`--fp8_base`, `--int8_convrot_base`, and the block-swap options are available here too. Output is a synchronized H.264/AAC MP4
with a JSON sidecar recording prompt, geometry, schedule, LoRA names, timings, and memory peaks.

The released weights are CFG-distilled: inference runs one evaluation per step with no negative-prompt branch.

## Training dashboard

> [!WARNING]
> Review every generated command before starting a long or expensive job. Dashboard contributed by [@Ada123-a](https://github.com/Ada123-a) in [PR #112](https://github.com/AkaneTendo25/musubi-tuner/pull/112).

The dashboard generates and runs H3 caching, training, and inference commands, and shows live output, metrics, and samples.

Install the dashboard Python dependencies. Release checkouts include the prebuilt frontend, so users do not need Node.js or npm:

```bash
pip install -e ".[dashboard]"
```

Start it from the repository root and open <http://127.0.0.1:7860>. `--host` defaults to `0.0.0.0`, so pass `--host 127.0.0.1`
explicitly to keep it local:

```bash
python -m musubi_tuner.gui_dashboard --host 127.0.0.1 --port 7860
```

On Windows, run `launch_musubi_dashboard.bat`, which passes `--host 127.0.0.1`. Create or load an H3 project, fill in the DiT, video/audio VAE, Qwen3-VL encoder, tokenizer, and dataset paths, review the generated commands, then cache and train. **Stop** requests a graceful shutdown so a state checkpoint can be written; keep `--save_state --autoresume` enabled to resume from the latest complete matching state.

> [!WARNING]
> The dashboard can launch arbitrary training processes with the permissions of its user. Do not bind it to a public interface unless access is protected. For remote use, keep `--host 127.0.0.1` and use an SSH tunnel.

Frontend developers can rebuild the bundled assets with `npm ci && npm run build` in `src/musubi_tuner/gui_dashboard/frontend`.
