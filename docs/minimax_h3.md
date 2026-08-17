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
- [Task contracts](#task-contracts)
- [Model download](#model-download)
- [Dataset](#dataset)
- [Pre-caching](#pre-caching)
- [Training](#training)
  - [Training a guidance-distilled model](#training-a-guidance-distilled-model)
  - [Full-parameter BF16 training](#full-parameter-bf16-training)
  - [Saving and resuming](#saving-and-resuming)
  - [Key options](#key-options)
  - [Memory and speed](#memory-and-speed)
  - [Training modes](#training-modes)
  - [Auxiliary objectives](#auxiliary-objectives)
  - [Validation](#validation)
- [Inference](#inference)
- [Training dashboard](#training-dashboard)

## Quick start

> [!TIP]
> Upstream [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) also supports MiniMax H3. Try the upstream implementation first unless you specifically need this fork's H3 extensions or dashboard.

1. Follow the upstream Musubi Tuner [installation instructions](https://github.com/kohya-ss/musubi-tuner#installation), including its Python and PyTorch requirements.
2. Prepare a TOML dataset using the shared upstream [dataset configuration guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md). The H3-specific task and media requirements are listed in [Task contracts](#task-contracts) and [Dataset](#dataset) below.
3. Configure Accelerate as described in the upstream [usage guide](https://github.com/kohya-ss/musubi-tuner#configuration-of-accelerate).
4. Download the H3 checkpoints, run [Pre-caching](#pre-caching), and then start [Training](#training). The [training dashboard](#training-dashboard) provides the same H3 caching, training, validation, sampling, and inference controls.

Common installation, dataset, Accelerate, and environment setup is intentionally not duplicated here. This page documents only the MiniMax H3 files, contracts, and commands that differ from upstream Musubi Tuner.

## Task contracts

Each training objective has a fixed dataset, conditioning-cache, and transformer contract. `--task` is passed to
`minimax_h3_cache_text_encoder_outputs.py`; the final column lists task-specific arguments for
`minimax_h3_train_network.py`. “None” means that the default trainer contract applies.

| Training objective | Example and dataset contract | Transformer | Cache `--task` | Trainer contract |
| --- | --- | --- | --- | --- |
| Text-to-image | [`image_fl2va.toml`](../examples/minimax_h3/image_fl2va.toml): `image_directory` or `image_jsonl_file` | FL2VA | `t2va` | None |
| First-image-conditioned image editing | [`image_fl2va_first.toml`](../examples/minimax_h3/image_fl2va_first.toml): one basename-matched control per target | FL2VA | `fl2va` | None |
| First+last-conditioned image editing | [`image_fl2va_first_last.toml`](../examples/minimax_h3/image_fl2va_first_last.toml): two ordered controls per target | FL2VA | `fl2va` | None |
| Text-to-video+audio | [`t2va.toml`](../examples/minimax_h3/t2va.toml): `video_directory` or `video_jsonl_file`; `h3_target_mode = "av"` is the default | FL2VA | `t2va` | None |
| Text-to-video only | [`video_only.toml`](../examples/minimax_h3/video_only.toml): video source plus `h3_target_mode = "video"` | FL2VA | `t2va` | None |
| Text-to-audio only | [`audio_only.toml`](../examples/minimax_h3/audio_only.toml): `audio_directory` or `audio_jsonl_file` plus `h3_target_mode = "audio"` | FL2VA | `t2va` | None |
| Video-to-audio | [`av.toml`](../examples/minimax_h3/av.toml): synchronized video source; `h3_target_mode = "av"` is the default | FL2VA | `t2va` | `--h3_observed_modality video` |
| Audio-to-video | [`av.toml`](../examples/minimax_h3/av.toml): synchronized video source; `h3_target_mode = "av"` is the default | FL2VA | `t2va` | `--h3_observed_modality audio` |
| First-frame image-to-video+audio | [`i2va.toml`](../examples/minimax_h3/i2va.toml): video source; first frame comes from the target | FL2VA | `i2va` | None |
| First+last-frame-to-video+audio | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml): video source; keyframes come from the target | FL2VA | `fl2va` | None |
| Last-frame image-to-video+audio | [`l2va.toml`](../examples/minimax_h3/l2va.toml): video source; last frame comes from the target | FL2VA | `l2va` | None |
| Fixed arbitrary references | [`ref2va.toml`](../examples/minimax_h3/ref2va.toml): target video, or [`image_ref2va.toml`](../examples/minimax_h3/image_ref2va.toml): target image; an audio target may carry references too; add `control_directory`, `control_path`, or numbered `control_path_N` | Ref2VA | `ref2va` | `--h3_training_mode ref2va` |
| Zero-or-more arbitrary references | [`ref2va_omni.toml`](../examples/minimax_h3/ref2va_omni.toml): target image or video; JSONL may omit references or use numbered `control_path_N` | Ref2VA | `ref2va_omni` | `--h3_training_mode ref2va_omni` |

For observed-modality objectives, the option names the modality supplied as clean **conditioning**, not the prediction target.
Consequently, `--h3_observed_modality video` defines video-to-audio training. See [Training modes](#training-modes) for the
corresponding noise and loss contracts.

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
│   └── minimax_h3_ref2va_bf16.safetensors                # Ref2VA only
├── text_encoders/
│   ├── qwen3vl_32b_minimax_h3_bf16.safetensors
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      # optional
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors
```

The tokenizer and processor metadata ship with the package, so `--tokenizer` is not needed. The loader validates which variant it
was given and never aliases an FL2VA checkpoint to Ref2VA.

To list a checkpoint's components, shards, tensor and parameter counts without allocating tensor storage:

```shell
python minimax_h3_generate_video.py \
  --model /models/MiniMax-H3 \
  --prompt "unused" --output unused.mp4 \
  --inspect
```

## Dataset

H3 uses Musubi's [shared dataset schema](./dataset_config.md) for video, image, and control fields. A target video's embedded soundtrack is the audio target; a video with
no audio stream trains as video-only with its audio loss masked. `control_directory` holds references whose basename matches
each target: controls for target `X` are `X.<ext>` or `X_<n>.<ext>`. A target whose own name ends in `_<n>` and that has no
direct match falls back to the shared prefix, but only over controls no other target claims; contested files are an error
rather than a silent reassignment. Relative `control_path`, `control_path_N`, `control_video_path_N`, and `control_audio_path_N`
values in a JSONL always resolve against that JSONL's directory. To combine separately stored video and audio as one
synchronized AV reference, use matching directories:

```toml
[[datasets]]
video_directory = "/data/targets"
control_video_directory = "/data/reference_video"
control_audio_directory = "/data/reference_audio"
control_modality = "av"
cache_directory = "/data/cache"
target_frames = [33]
```

The three directories must contain basename-matched files. Set `control_modality = "video"` to omit reference soundtracks or
`control_modality = "audio"` to retain only reference audio. For multiple references matched through `control_directory`, use an
ordered `control_modalities = ["video", "audio", "av"]` list. The choice is part of latent and text caching; recache both after
changing it. Ref2VA requires at least one visual reference, so an audio-only entry must accompany an image or video reference.

Both caches record which reference files produced them, so swapping, reordering, or editing a reference rebuilds that item under
`--skip_existing`. Reference caches written before this identity was recorded are rebuilt once.

For modality dropout, replace the static modality setting with probabilities in `[av, video, audio]` order:

```toml
control_modality_probabilities = [0.5, 0.25, 0.25]
```

The text cache stores every enabled presentation and training draws one mode per item. The same draw is reused by the trainable,
guidance, and preservation forwards. Video-only removes reference soundtracks; audio-only retains image references as visual
anchors and uses the audio from paired AV references. Recache text outputs after changing the probabilities. Latent caches do not
need to be rebuilt. Every enabled mode must leave at least one image or video reference, as required by the released Ref2VA
model, and this is checked when the dataset config is parsed: a nonzero `video` weight needs an image or video reference, and a
nonzero `audio` weight needs an image reference, since video references become audio in that mode.

Reference audio is cropped or zero-padded to the canonical sample count of its reference video span (`temporal_shape`), the same
grid the target audio uses. Reference rows carry no validity mask, so padding added to a short reference track is
indistinguishable from silence to the model; keep reference audio at least as long as its reference video.

To restrict video or image loss to selected regions, set `loss_mask_directory` to a directory of masks with matching target
basenames, or set `default_loss_mask_path` as a fallback. A JSONL item may override either with `loss_mask_path` (the alias
`video_loss_mask_path` is also accepted). A mask may be a still image, video, or frame directory; still images repeat across the
clip and shorter mask sequences repeat their last frame. White pixels contribute to loss and black pixels do not.
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
video_directory = "/path/to/videos"
cache_directory = "/path/to/cache"
target_frames = [124, 175, 243, 294, 362]
frame_extraction = "uniform"
```

`h3_target_mode` selects which modalities are packed at all:

| Value | Effect |
| --- | --- |
| `av` (default) | video and audio |
| `video` | omits audio decoding, caching, and rows |
| `audio` | audio only; needs `audio_directory` or `audio_jsonl_file`, and reuses `target_frames` for duration |

An audio-only dataset needs **exactly one** `target_frames` value, and it must be on the `17k+5` grid — the multi-value list in
the example above is video-only. `audio_directory` takes same-stem `.txt` captions; `audio_jsonl_file` takes records with
`audio_path` and `caption`.

An audio target may also declare Ref2VA references, with the same fields a video target uses: `control_directory`,
`control_video_directory` + `control_audio_directory`, or per-record `control_path_N` / `control_video_path_N` +
`control_audio_path_N` / `control_modality_N` in the audio JSONL. That trains video-to-audio with an **arbitrary** conditioning
video (Foley), or audio generation from a reference voice clip plus a visual anchor. Ref2VA still requires at least one image or
video reference, so a reference audio clip must accompany a visual one. Cache with `--task ref2va` and train with
`--h3_training_mode ref2va`; the packed sequence is `[text | references | target audio]` and no target video rows are emitted.
Reference video length follows the target's `target_frames`, and the RoPE spatial grid comes from the dataset `resolution` (the
same geometry an audio-only cache already records). Because the references are visual, `--vae` is then required for latent
caching even though the target has no video. Audio-target caches are named `<stem>_audio<hash>_…`, where the hash covers the absolute source
path, so an audio file never overwrites a video cache of the same stem and two same-stem audio files from different directories
stay apart in one `cache_directory`. Audio caches written before this naming are rebuilt once.

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

Omit `--audio_vae` for image-only or video-only datasets; omit `--vae` for audio-only datasets without visual references. Add `--cache_guidance_empty` if you plan to
use caption dropout or the guidance objective.

`--task` must match how you intend to train: `t2va` (text only), `i2va` (first frame), `fl2va` (first+last), `l2va`
(last frame), `ref2va`, or `ref2va_omni`. Keyframe tasks take their frames from the target video itself, not from control fields.

For FL2VA conditioned-image training, pass the same `--h3_image_mode first` or `--h3_image_mode first_last` to both cache
commands and use `--task fl2va` for text caching. `h3_image_frame_count` in the dataset, or the CLI override
`--h3_image_frame_count`, selects a `17k+5` target grid and defaults to 5. A still target repeats across that grid;
`multiple_target = true` instead resamples an ordered target-image sequence. The latent and text caches carry the same
source fingerprint, so changed/reordered targets or controls are rebuilt under `--skip_existing` and mismatched cache pairs
are rejected during training. `--h3_text_visual_max_pixels` limits only the images presented to Qwen3-VL.
`--h3_max_caption_tokens N` optionally truncates only the caption while retaining every structural image/video token. Its
default `0` leaves captions unchanged; pass the same nonzero value to text caching and training.

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

To reduce conditioner VRAM, add `--text_encoder_quantization int8` or `nf4` to the BF16 checkpoint, or load the released
pre-quantized file with `--text_encoder_quantization nvfp4_awq`:

```shell
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --text_encoder_quantization nvfp4_awq \
  --task t2va --device cuda
```

All three are precision trade-offs, not equivalents. They reduce GPU residency only: the BF16 checkpoint is still memory-mapped
into host address space while weights are converted, so the host RAM requirement is unchanged.

If the conditioner still does not fit, `--h3_text_encoder_blocks_to_stream N` keeps `N` of its 50 frozen language layers in
CPU memory and moves them through a fixed two-layer GPU ring during encoding. `50` minimizes weight residency; smaller values
trade less transfer overhead for less memory saved. This is opt-in, requires CUDA, and currently supports BF16 and native
`nvfp4_awq` checkpoints. It does not support the bitsandbytes `int8` or `nf4` loader.

On a Blackwell GPU with PyTorch 2.10 or newer, `--h3_nvfp4_scaled_mm` additionally quantizes Qwen activations to FP4 and
uses the hardware W4A4 matrix kernel. It applies only to a native `nvfp4_awq` conditioner and fails early on unsupported
hardware instead of changing execution silently. Leave it off when conditioning fidelity matters most: the default path
keeps BF16 activations and uses NVFP4 only for stored weights.

## Training

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

Adapters target attention and feed-forward projections; norms and timestep/modality calibration stay frozen. LoHa/LoKr are
unsupported. Regional `torch.compile` covers all 50 main blocks and both text-refiner blocks; use `--compile` and optionally
`--compile_auto_cache_size_limit`, `--compile_fallback_to_eager`, or `--inductor_config KEY=VALUE ...`. GPU compilation requires
a working Triton installation; on Windows, install a `triton-windows` build compatible with the installed PyTorch and Python
versions.

### Training a guidance-distilled model

H3 is guidance-distilled, so direct LoRA training can be inefficient or alter its CFG-free, few-step behavior. Two optional
strategies address different goals:

1. `--h3_base_preservation_loss_weight 0.02` limits drift from the frozen base. Add
   `--h3_base_preservation_probability 0.25` to evaluate it on 25% of batches with inverse-probability loss scaling. This
   preserves the expected loss, but rare larger updates are not optimizer-equivalent to applying the dense loss every step.
2. If a compatible de-distillation training adapter is provided, load it through `--base_weights` while training the concept
   LoRA, then remove it for inference. Adapters are checkpoint-specific: the Ref2VA checkpoint needs its own adapter, not one
   made for FL2VA. One community example is
   [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter), whose
   `minimax_h3_ref2va_training_adapter_v1.safetensors` targets the Ref2VA checkpoint.

For concept LoRA training over a de-distillation adapter, sparse preservation can provide an additional anchor, but the two
objectives are not equivalent: preservation retains the loaded base's predictions, while a de-distillation adapter deliberately
changes them. The adapter remains loaded only during LoRA training; validate the resulting concept LoRA against stock H3.

The training adapter is an approximation. The community does not have MiniMax's original undistilled teacher, so no adapter can
reliably reconstruct the ideal undistilled prediction for every prompt, timestep, modality, and concept. Validate short runs and
check samples again after removing the adapter.

### Full-parameter BF16 training

`minimax_h3_train.py` updates the entire transformer and writes a native MiniMax H3 BF16 checkpoint. It is separate from the
LoRA entry point above. Full training requires the ordinary BF16 FL2VA or Ref2VA checkpoint; FP8, ConvRot INT8, pruned AdaLN,
LoRA initialization/merge weights, and architecture-changing options are rejected.

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
| `--blocks_to_swap N` | Use backward-capable block swap for trainable weights. Increase `N` when the model or activations do not fit; unlike LoRA block swap, do not add `--block_swap_h2d_only`. |
| `--block_swap_trainable_ring` | Use coalesced bidirectional block transfers and write updated weights back to pinned CPU masters. Requires block swap, gradient checkpointing, fused backward, and `--use_pinned_memory_for_block_swap`. |
| `--block_swap_ring_size N` | Number of reusable GPU block buffers for the trainable ring; `2` enables double buffering. |
| `--gradient_checkpointing_cpu_offload` | Offload checkpoint activations when long packed sequences still exceed VRAM. |
| `--mem_eff_save` | Stream native transformer tensors during `.safetensors` output; enabled by default. |
| `--no_mem_eff_save` | Write checkpoints with the ordinary safetensors writer instead of the streaming one; needs the whole checkpoint contiguous in host memory. |

For a 24 GB-class GPU, start with `--blocks_to_swap 48`, the trainable ring, and `--block_swap_ring_size 2`. Keep activation
CPU offload disabled initially so checkpoint recomputation stays on the GPU; add `--gradient_checkpointing_cpu_offload` only
if the chosen resolution, frame length, or conditioning mode still exceeds available VRAM.

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

The recommended command above uses `--save_state --autoresume`: every periodic checkpoint gets a resumable `*-state`
directory, and restarting the same command automatically selects the highest-step complete state matching `--output_name`.
More generally, add `--save_state` alongside `--save_every_n_steps N` or `--save_every_n_epochs N` to create resumable states.
Resume by pointing `--resume` at that directory, not at its neighboring `.safetensors` LoRA file:

```shell
--resume output/h3_style-step00001000-state
```

`--resume` restores the optimizer, scheduler, dataloader position, RNG streams, epoch, and displayed global step.
`--network_weights output/h3_style-step00001000.safetensors` only initializes LoRA weights for a new run; its optimizer,
scheduler, and step start at zero. Use `--save_state_on_train_end` if only the final state is needed. An incomplete state is
rejected instead of silently restarting at step zero. Add `--autoresume` to select the highest-step complete state matching
`--output_name` in `--output_dir`; an explicit `--resume` path takes priority.

Add `--async_checkpoint_save` to keep periodic saves off the training thread: the loop pauses only long enough to copy the
LoRA weights to CPU, and hashing plus the file write finish in the background, so the file appears shortly after the step.

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
| `--sdpa`, `--flash_attn`, `--flash3` | `--sdpa` | Attention backend. Each FlashAttention flag needs its package, and `--flash3` needs a Hopper GPU. Both fall back to SDPA on padded batches. |
| `--h3_attn_auto_dispatch` | off | Prefer cuDNN SDPA for large maskless workloads. Changes rounding; benchmark first. |
| `--h3_int8_attention {off,aux,train}` | `off` | Experimental native INT8-QK forward with BF16/FP16 P×V and an optimized training backward. `aux` affects only guidance and base-preservation teacher forwards; `train` also affects the trainable forward. Requires CUDA, Triton, and head width 128; masked or padded batches use the selected regular backend. Incompatible with `--compile`. |
| `--h3_lora_token_refiner` | off | Also place LoRA adapters on the two text token-refiner blocks. This experimental target can strengthen trigger or identity binding, adds eight adapter modules, and otherwise leaves the existing main-block targets unchanged. |
| `--compile` | off | Regionally compile all H3 blocks with the selected backend/mode. Compatible with full or partial gradient checkpointing and with block swap; swapped Linear calls stay eager. |
| `--h3_fused_qk_norm_rope` | off | Use the custom Triton Q/K RMSNorm+RoPE kernel outside compiled graphs. It is faster but changes BF16 rounding, so it is opt-in. |
| `--h3_fused_indexed_adaln` | off | Fuse main-block RMSNorm with token-indexed AdaLN shift/scale for a frozen LoRA base. Requires CUDA and Triton; unsupported layouts, trainable norm/AdaLN parameters, and compiled blocks use the regular path. It changes BF16 rounding and is most useful with gradient checkpointing. |
| `--h3_fused_swiglu` | off | Fuse the SwiGLU activation in main and token-refiner feed-forward layers. Requires CUDA and Triton; unsupported layouts and compiled blocks use the regular path. It changes BF16 rounding, so benchmark and validate it before a long run. |
| `--h3_swiglu_chunk_rows N` | `0` | Process each main-block feed-forward layer in sequence-row chunks to reduce peak VRAM. Start with `2048`; smaller values may save more memory but add overhead. Incompatible with `--compile`. |
| `--h3_gradient_checkpointing_cpu_offload_pin_memory` | off | Pin CPU-offloaded checkpoint activations for faster transfers. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload` and substantial free system RAM. |
| `--h3_reusable_activation_offload` | off | Reuse pinned CPU checkpoint buffers and prefetch activations in reverse block order. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload` and sufficient free system RAM. |
| `--gradient_checkpointing_cpu_offload_dtype` | `none` | Wire dtype of the offloaded activations. `fp8_e4m3` quantizes each large bf16/fp16 activation on the GPU (per-tensor scale) before the D2H copy, halving PCIe traffic and pinned host memory; the recomputation then consumes a lossy activation (e4m3 keeps three mantissa bits, roughly 3% relative resolution), so gradients differ from an exact run (deterministically, but not bit-identically). What the offloader stores is one block-boundary hidden state per checkpointed block, and re-running a block from a slightly perturbed input is well conditioned: measured gradient error stays around 1-2% of the gradient's own magnitude. Payloads under 1 MiB, integer/boolean saves, and fp32 activations pass through unquantized. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload --h3_reusable_activation_offload`. Use it when long Ref2VA sequences make pinned host memory or PCIe bandwidth the bottleneck; keep `none` for exactness-sensitive runs. |
| `--h3_gradient_checkpointing_blocks N` | all 50 | Checkpoint only the last N main blocks. This explicit speed/VRAM trade-off requires `--gradient_checkpointing` and resident eager blocks. It can be combined with `--compile` (the eager blocks add one extra compiled variant) but not with block swap, whose streamed weights are only safe to reuse at recompute time. Each eager block retains its full activations, which at video sequence lengths costs several GB per block; on 80 GB, video training fits only a few eager blocks. |
| `--h3_shift_video` / `--h3_shift_audio` | `12.0` / `3.0` | Released per-modality flow shifts. Both derive from one shared coordinate; keep the defaults unless reproducing a measured experiment. |
| `--timestep_sampling` | `uniform` | Shape of the shared unshifted schedule. Keep `uniform` unless deliberately testing a different distribution. `sigmoid`, `logsnr`, and `sigma` are experimental alternatives; `shift` with H3's required `--discrete_flow_shift 1` is equivalent to sigmoid sampling. Model-specific dynamic-shift modes are rejected because H3 applies its own video/audio shifts afterward. |
| `--weighting_scheme` | `none` | Optional loss weighting. H3 bounds `sigma_sqrt` inverse-square weighting so that a near-zero draw cannot dominate a batch-1 optimizer step; `cosmap` is intrinsically bounded. |
| `--h3_sigma_sqrt_max_weight` | `10` | Maximum per-sample weight used by `sigma_sqrt`. It preserves extra emphasis at low sigma without the generic unbounded spike and has no effect with the default `none` scheme. |
| `--discrete_flow_shift` | `1.0` | Must stay at the default; H3 applies its own shifts. |
| `--h3_image_flow_shift` | auto | Fixed shift for image batches only. |
| `--h3_timestep_focus_probability` | `0` | Optional video/AV curriculum: draw this fraction of the uniform base schedule from `--h3_timestep_focus_min` through `--h3_timestep_focus_max` (defaults `0.4` and `0.8`) while retaining full-range samples. Image batches keep their resolution-aware schedule. Requires `uniform` and cannot be combined with min/max timestep clipping. |
| `--num_timestep_buckets` | off | Stratifies base timesteps across an epoch to reduce sampling imbalance; it does not change the intended distribution or make a step faster. It is incompatible with `--timestep_sampling sigma`. |
| `--h3_video_loss_weight` / `--h3_audio_loss_weight` | `1.0` | Modality weights. `--h3_loss_balance token` switches from equal modality means to element weighting. |
| `--async_checkpoint_save` | off | Hash and write periodic checkpoints on a background thread. The step pauses only for the CPU snapshot, so the `.safetensors` file lands shortly after the step instead of stalling training for the write. Saves stay ordered and never overlap; the final checkpoint and `--save_state` remain synchronous. |

### Memory and speed

Quantize the frozen base, reduce AdaLN, then swap blocks — in that order.

```shell
  --h3_convrot_int8 --h3_convrot_int8_fwd bf16 --h3_adaln_rank 16
```

This is the recommended configuration. ConvRot INT8 stores the frozen linear weights at one byte per value,
`--h3_convrot_int8_fwd bf16` evaluates those weights with BF16 activations, and rank 16 replaces the full AdaLN projections with
compact factors.

| Option | Purpose |
| --- | --- |
| `--h3_convrot_int8` | Quantize the released BF16 checkpoint to ConvRot INT8 at load. Rejects `--fp8_base` and `--int8_convrot_base`. |
| `--h3_convrot_int8_fwd bf16` | Recommended. Evaluates the matmul in BF16 without changing stored weights. |
| `--h3_convrot_int8_bwd int8` | INT8 input-gradient path for GPUs without FP8 support. |
| `--h3_convrot_int8_lora_fused` | Fuse LoRA application into supported ConvRot INT8 linear calls. Requires an INT8 ConvRot base (online or pre-quantized), `--h3_convrot_int8_fwd int8`, `--h3_convrot_int8_bwd int8`, and triton. |
| `--fp8_base` | Scaled FP8. `--h3_fp8_quantization_mode` selects `block` (default, fastest), `channel`, or `tensor`. |
| `--int8_convrot_base` | Load the released pre-quantized checkpoint instead of quantizing at load (see below). |
| `--h3_adaln_rank 16` | Reduce the AdaLN projections, the largest parameter group (13.0B of 33.1B), to ~77M. |

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
pre-quantized base. Its compact AdaLN projections are already baked into the checkpoint, so do not pass `--h3_adaln_rank`.

Generic LoRAs passed through `--base_weights` are merged while H3 loads. With a pre-quantized ConvRot base, only affected layers
are dequantized, all requested LoRAs and `--base_weights_multiplier` values are accumulated in FP32, and each layer is
requantized once. The resulting frozen base is then used for ordinary LoRA training.

`--h3_adaln_rank` avoids quantizing the AdaLN projections and composes with either quantization mode. It is rejected on
already-pruned and INT8 ConvRot checkpoints.

Block swapping streams frozen weights from host memory. It is valid only while the base is frozen:

```shell
  --blocks_to_swap 40 \
  --block_swap_h2d_only \
  --block_swap_ring_size 2
```

`--use_pinned_memory_for_block_swap` can improve transfer bandwidth when the host has enough available memory; leave it disabled
when pinned allocations stall or fail. `--block_swap_granularity layer` streams individual `Linear` layers through the same
H2D-only ring and supports all 50 blocks, at the cost of more transfers; use the default `block` granularity when it fits. It
cannot be combined with `--h3_convrot_int8` or `--int8_convrot_base`. Add
`--gradient_checkpointing_cpu_offload` when sequence length would otherwise exceed VRAM, and set
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

For low host RAM and low VRAM, start with the released BF16 checkpoint and let the loader reduce and quantize weights while
placing swapped blocks on CPU:

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

If this configuration exceeds available VRAM, raise `--blocks_to_swap` to `50` and use `--block_swap_ring_size 1`; those
settings trade throughput for lower device residency. `--block_swap_granularity layer` is rejected here because it bypasses the
ConvRot INT8 forward. Block-swap loading materializes CPU-master copies only for swapped blocks, so
`--blocks_to_swap` affects both host and device residency. Total training memory also depends on packed sequence length,
attention workspaces, adapter rank, gradients, and optimizer state.

Use `--h3_gradient_checkpointing_cpu_offload_pin_memory` only when the host has substantial free RAM. Long Ref2VA sequences can
require substantial pinned memory in addition to model-loading and dataset memory. When that pinned footprint is the binding
constraint, `--h3_reusable_activation_offload --gradient_checkpointing_cpu_offload_dtype fp8_e4m3` halves it.

DataLoader tuning affects cache delivery, not transformer compute. Increase `--max_data_loader_n_workers` only when storage or
CPU loading leaves the GPU idle; `--persistent_data_loader_workers` avoids restarting those workers each epoch, and
`--dataloader_prefetch_factor N` controls queued batches per worker. `--dataloader_pin_memory` enables pinned batch buffers and
non-blocking device copies, but consumes locked host RAM. These options are off/default-controlled unless explicitly set and
normally do not speed up an H3 run whose cached batches already keep the GPU busy.

### Training modes

These modes use target-derived conditioning and therefore use a `t2va` conditioning cache, except where the table below says
otherwise: masking and `per_row_sigma` extension only pin rows inside the target block, so they also combine with
`--h3_training_mode ref2va` / `ref2va_omni` and their reference caches. For observed-modality training,
`h3_target_mode = "av"` is required and each target video must contain its synchronized soundtrack — an audio-only dataset has no
video rows to observe, so `--h3_observed_modality` cannot be used with it. Conditioning an audio target on video is still
possible through Ref2VA references instead, which supply an arbitrary conditioning video rather than the target's own track; see
[Dataset](#dataset). The observed modality remains in the packed attention sequence but its
loss weight is forced to zero; this isolates direct supervision, not H3's shared attention parameters.

| Option | Trains |
| --- | --- |
| `--h3_observed_modality {video,audio,random}` | Video-to-audio, audio-to-video, or one adapter covering both plus joint |
| `--h3_extension_video_frames N` / `--h3_extension_audio_latents N` | Continuation from an observed prefix. Counts are in **latent** units and each must be shorter than its target; the two are independent, so setting one leaves the other generated in full. Under Ref2VA only the `per_row_sigma` route is supported |
| `--h3_extension_probability P` | Train the extension recipe on a synchronized random fraction of steps; the rest train the plain objective. Requires the extension flags. `1` (default) applies it every step |
| `--h3_keyframe_anchors first,11,last` / `--h3_keyframe_random_count N` | Interpolation from arbitrary anchors. FL2VA/`t2va` caches only |
| `--h3_mask_mode {box,border,segment,dataset}` | Inpainting, outpainting, temporal infilling from a procedural mask, or the region the dataset authored. Also available under Ref2VA |
| `--h3_mask_probability P` | Train the masked recipe on a synchronized random fraction of steps; the rest train the plain objective. Requires `--h3_mask_mode` or `--h3_mask_audio`. `1` (default) applies it every step |
| `--h3_frame_sigma_jitter 0.2` | Spreads target-frame noise levels across the schedule in one step. Supported by native T2VA/I2VA/FL2VA/L2VA/Ref2VA caches, including guidance-consistent loss, and skipped for images; cannot be combined with in-target observed-row options or sigma-dependent loss weighting; `0` disables it |
| `--h3_spatial_density_jitter 0.2` | Perturbs the area normalization of the spatial RoPE grids each step, drawn log-uniformly from `[1/1.2, 1.2]`, so fixed-resolution data still trains a range of token spacings. One factor covers every grid in the packed sequence; `0` disables it |
| `--h3_caption_dropout_rate 0.1` | Trains the unconditional branch; requires `--cache_guidance_empty` |

Extension, keyframes, and masking all claim the observed rows, so **only one may be active on a step**.

**Mixing recipes.** `--h3_mask_probability` and `--h3_extension_probability` turn the choice into one categorical draw per
optimizer step — shared by every item of the batch, by the guidance and preservation branches, and across distributed ranks — so a
single run can teach masking, extension and the plain objective. Setting both allows masking and extension in the same run,
provided the two probabilities sum to at most `1`; the draw selects at most one of them per step and the remainder trains the plain
objective. Without probabilities the two remain mutually exclusive. Keyframes stay exclusive either way. A step that does not
select a recipe builds exactly the tensors of a run without those flags: unlike guidance distillation and base preservation, no
loss is rescaled by its probability, because mixing trains different objectives on different steps rather than a sparse estimate
of one — the same convention as caption dropout and `--h3_observed_modality random`. Validation keeps measuring one fixed recipe
(the masked one in a mixed run) so its numbers stay comparable. `h3/recipe_mask_active` and `h3/recipe_extension_active` are
logged only once a probability below `1` is set.

**Observed modality.** The observed side is pinned to the noise level the release already uses for conditioning of that kind, and
carries no training signal — its loss weight is forced to zero. Datasets must cache both modalities. `random` redraws the task
each step; validation still reports the joint objective so its numbers stay comparable.

**Extension.** `--h3_extension_route` chooses the presentation: `condition_rows` (default) duplicates the context as clean rows,
matching the released keyframe contract but costing sequence length; `per_row_sigma` pins it in place with no extra tokens, at the
cost of intra-block noise levels the released weights have not seen. The observed span is removed from the loss. Under
`--h3_training_mode ref2va` / `ref2va_omni`, `condition_rows` is rejected: duplicating the context needs packer support the
Ref2VA layout does not have, so pass `--h3_extension_route per_row_sigma`.

**Keyframes.** Entries are `first`, `last`, or a latent frame index. Anchors stay in the loss, matching the released contract.
`last` is the final *pixel* frame, deliberately not the same anchor as the integer `frames - 1`. Keyframes remain FL2VA-only;
they add condition rows that only the T2VA packer can place.

**Masking.** Masks are drawn per step, so the occlusion distribution changes without re-caching. Masks combine with every
training mode: under Ref2VA the observed rows are pinned inside the target block, the reference rows keep their own conditioning
noise level, and neither is scored. `--h3_mask_min_fraction` and
`--h3_mask_max_fraction` bound the masked fraction; `--h3_mask_audio` also hides a run of audio latents. Masks are reduced to the
`(1, 2, 2)` patch grid and a patch counts as generated when any latent inside it is, so at small latent resolutions a wide
fraction range can leave nothing observed.

A conditioning mask is not a loss mask: it says which pixels the model *observes* as clean context, while `loss_mask_directory`
and friends say which pixels are *scored*. Both may be set; the effective loss is their intersection.

**Dataset masks.** `--h3_mask_mode dataset` reads the observed region from the dataset instead of drawing one, for segmentation
maps, mattes, or hand-drawn regions.

| Field | Behavior |
| --- | --- |
| `conditioning_mask_directory` | Directory of mask images matched to targets by basename, exactly one per target |
| `conditioning_mask_path` | Per-record override in an image or video JSONL, resolved against the JSONL's directory |

Masks are still images (`png`, `jpg`, …), read as luma: pixels above `127` are **observed context**, everything else is the
region to generate. The same plane applies to every frame — video masks are not supported this round. The mask is resized to the
item's bucket resolution with nearest-neighbour sampling and no antialiasing (aspect ratio is not preserved, since a mask is
authored against its target), then reduced to latent cells: a cell is observed only when every pixel inside it is, which keeps
the region boundary on the generated side just as the patch reduction does. Masks are applied to the packed rows at train time,
so they never enter the latent cache and never invalidate it — adding, editing, or removing one needs no re-cache.

Downstream the behavior is identical to a procedural mask: observed rows are pinned clean at the conditioning noise level and
excluded from the loss, under `t2va`, `ref2va` and `ref2va_omni` alike, and `--h3_mask_probability` mixes it with the plain
objective per step as usual. `--h3_mask_min_fraction` / `--h3_mask_max_fraction` do not apply — the region is authored, not
drawn — but a mask that reduces to all-observed or all-generated is warned about once, naming the item. `--h3_mask_audio` stays
procedural: dataset masks are video-only, and declaring one on an audio dataset is an error. Because the observed rows are shared
by the whole batch, a batch whose items carry different masks is rejected; use `batch_size = 1` for such datasets. A config that
declares masks under a procedural mode, or selects `dataset` without them, is rejected when the dataset config is parsed.

```toml
[[datasets]]
video_directory = "/data/clips"
conditioning_mask_directory = "/data/masks"   # clip_01.mp4 -> clip_01.png, white = observed
batch_size = 1
```

### Auxiliary objectives

Guidance distillation and active base preservation add no-gradient transformer forwards over the packed sequence, so their cost
grows with sequence length and is larger for reference-conditioned batches. CREPA reuses features from the main forward and does
not add a complete H3 forward.

| Option | Purpose |
| --- | --- |
| `--h3_guidance_distillation_scale 4` | Guidance-consistent objective using cached empty-text conditioning. A scale of `4` is recommended; `3` is generally too weak. `--h3_guidance_loss_form` selects `normalized` or `contrastive`; both share an optimum, but contrastive is `scale²` larger. |
| `--h3_guidance_distillation_probability 0.5` | Evaluate the empty-conditioning branch on a synchronized random fraction of batches, skipping its extra forward on the rest, and scale the guidance correction by `1 / probability`. `1` (default) applies the objective every batch; smaller values preserve the expected loss, but rare larger corrections are not optimizer-equivalent to applying the dense objective every step. |
| `--h3_guidance_loss_schedule {sigma,constant}` | `sigma` (default) scales guidance from `1` at the clean endpoint to the configured value at maximum noise, independently for video and audio. `constant` retains the configured scale everywhere. |
| `--h3_base_preservation_loss_weight 0.02` | Recommended starting value. Penalizes drift from the frozen base's prediction and anchors to whichever base is loaded, quantized or not. |
| `--h3_base_preservation_probability 0.25` | Evaluate preservation on a synchronized random fraction of batches and scale active losses by `1 / probability`. `1` applies the objective every batch; `0.25`–`0.5` is a faster approximation whose rare scaled updates interact differently with clipping and adaptive optimizers. |
| `--crepa` | Temporal representation alignment for video training. |

Treat `--h3_base_preservation_loss_weight 0.02` as an initial value rather than a universal setting. Its effect depends on
training length, quantization, adapter rank, dataset, and learning rate. Higher values such as `0.05` can preserve the base very
strongly but substantially slow concept learning. Monitor training and validation samples, and reduce or disable the objective
when preservation dominates. The value can be changed when resuming training.

CREPA aligns projected features from an earlier block with a later block (`mode=backbone`) or with frozen DINOv2 features
(`mode=dino`). Only generated video rows participate; image, audio-only, and video-observed batches are skipped.

```shell
accelerate launch minimax_h3_train_network.py ... \
  --crepa mode=backbone student_block=16 teacher_block=33 weight=0.05 tau=1 neighbors=2
```

| Field | Default | Meaning |
| --- | ---: | --- |
| `mode` | `backbone` | `backbone` uses a later H3 block as teacher; `dino` uses cached DINOv2 features. |
| `student_block` / `teacher_block` | `16` / `33` | Zero-based blocks; student must precede teacher, teacher below 50. |
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
training requests.

CREPA's projection head is training state, not part of the inference LoRA; it is saved separately as `h3_crepa.safetensors`.

### Validation

```shell
accelerate launch minimax_h3_train_network.py ... \
  --validation_dataset_config validation.toml \
  --validate_at_start --validate_every_n_steps 100 \
  --validation_timestep_bins 4 \
  --validation_min_timestep 100 --validation_max_timestep 900 \
  --max_validation_items 8
```

The validation TOML uses the same format and prebuilt caches as training, with `batch_size = 1`. Evaluation is deterministic —
each index and bin draws stable noise from `--validation_seed`, defaulting to the training seed — and restores RNG state, so
enabling it does not change training randomness.

Bins are equal-width midpoints between `--validation_min_timestep` and `--validation_max_timestep` in the shared *unshifted*
coordinate; video and audio then receive their own shifts, and image batches use the same `--h3_image_flow_shift` or automatic
image shift as image training. TensorBoard receives `val/loss`, `val/loss/video`, `val/loss/audio`, and `val/loss/bin_NN`. CREPA and base-preservation are excluded, since they are regularizers rather than
held-out reconstruction quality.

When generated or reference modalities are randomized during training, validation evaluates every enabled task deterministically
and prefixes metrics with `joint`, `v2a`, `a2v`, and, when applicable, `ref_av`, `ref_video`, or `ref_audio`. This adds validation
forwards only for the extra tasks; ordinary fixed-task validation keeps its original cost.

### Sampling during training

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
| `--keyframe INDEX:PATH` | Keyframe at an arbitrary latent frame, repeatable. |
| `--reference_image` / `--reference_video` / `--reference_audio` | Ref2VA references; audio must accompany an image or video. A reference video's own soundtrack is included automatically, so do not also pass it as `--reference_audio`. Requires the Ref2VA checkpoint. |
| `--reference_image_size_mode` | Ref2VA image sizing: `short_edge` keeps the released behavior; `target_area` preserves aspect ratio and uses approximately the target output area. |
| `--reference_image_short_edge` | Reference-image short edge in `short_edge` mode (default 2048). |
| `--reference_image_max_pixels` | Optional pixel-area cap in `target_area` mode; `0` uses the target output area. |
| `--reference_video_short_edge` | Reference-video short edge (default 768, minimum 32). Lower values reduce Ref2VA reference rows, speed cost, and VRAM. Use the same value for latent caching, text caching, training, and inference. |
| `--reference_video_max_pixels` | Maximum pixels per reference-video frame after aspect-preserving resize (default 768×1344, minimum 1024). The cap is enforced on the final 32-aligned dimensions, so extreme aspect ratios are downscaled rather than rounded back over it. Use the same value at every stage. |
| `--reference_video_fps` | Caching and training only (no inference flag yet). Subsample every reference video to this many frames per **source** second so the whole clip conditions the model instead of only its opening span. `0` (default) keeps the released behavior of truncating the reference to the target's frame count, bit for bit. Frame `k` is source frame `round(k × source_fps / F)` (half up, deterministic), and the result is snapped to the nearest legal reference length (`1` or `17n+5`) that still fits the target's frame budget, padding with the final frame when it lands short. A reference video's paired soundtrack cannot be subsampled without pitch artifacts, so the asymmetry is deliberate: the subsampled reference video spans the whole clip, while its paired soundtrack still covers the clip's opening span. Changing the value changes the reference latents, so it enters the cache identity: pass the same value to `minimax_h3_cache_latents`, `minimax_h3_cache_text_encoder_outputs`, and training, and re-cache both stages when it changes. |
| `--lora_weight` / `--lora_multiplier` | Attach saved adapters. |
| `--steps` | Sigma grid points including terminal zero, so `20` runs 19 evaluations. |

For a conditioned still image, select `--h3_image_mode first` with `--first_frame`, or `first_last` with both endpoint
images, and use an image extension for `--output`. The default 5-frame grid is decoded through the video VAE and
`--h3_select_frame` chooses the saved frame; the audio VAE is not required for image-file output.

`--fp8_base`, `--int8_convrot_base`, and the block-swap options are available here too. Output is a synchronized H.264/AAC MP4
with a JSON sidecar recording prompt, geometry, schedule, LoRA names, timings, and memory peaks.

The released weights are CFG-distilled: inference runs one evaluation per step with no negative-prompt branch.

## Training dashboard

> [!WARNING]
> The dashboard is a proof of concept and is not guaranteed to be maintained. Review every generated command before starting a long or expensive job. The original dashboard was created by [@Ada123-a](https://github.com/Ada123-a) in [PR #112](https://github.com/AkaneTendo25/musubi-tuner/pull/112).

The H3 web dashboard creates project configurations, runs H3 caching, training, and inference commands, shows live process output and training metrics, previews samples, and stops training gracefully.

Install the dashboard Python dependencies. Release checkouts include the prebuilt frontend, so users do not need Node.js or npm:

```bash
pip install -e ".[dashboard]"
```

Start it from the repository root and open <http://127.0.0.1:7860>:

```bash
python -m musubi_tuner.gui_dashboard --host 127.0.0.1 --port 7860
```

On Windows, run `launch_musubi_dashboard.bat`. Create or load an H3 project, fill in the DiT, video/audio VAE, Qwen3-VL encoder, tokenizer, and dataset paths, review the generated commands, then cache and train. **Stop** requests a graceful shutdown so a state checkpoint can be written; keep `--save_state --autoresume` enabled to resume from the latest complete matching state.

> [!WARNING]
> The dashboard can launch arbitrary training processes with the permissions of its user. Do not bind it to a public interface unless access is protected. For remote use, keep `--host 127.0.0.1` and use an SSH tunnel.

Frontend developers can rebuild the bundled assets with `npm ci && npm run build` in `src/musubi_tuner/gui_dashboard/frontend`. Node.js is only a development dependency.
