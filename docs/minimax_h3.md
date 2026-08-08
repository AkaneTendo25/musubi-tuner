# MiniMax H3

LoRA training and inference for MiniMax H3, a 50-block transformer that generates video and stereo audio jointly.

Media contract: 24 fps video on the `17k+5` frame grid, 24-channel latents at spatial compression 16 with a `(1, 2, 2)` patch;
32 kHz stereo audio, 32-channel latents at 40 latent frames per second; video and audio flow shifts 12.0 and 3.0.

Two released transformers, with different conditioning contracts:

| Checkpoint | Covers |
| --- | --- |
| **FL2VA** | text-to-video, first-frame I2V, first+last keyframes, video-only, audio-only, still images |
| **Ref2VA** | arbitrary image, video, and audio references |

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

H3 uses Musubi's standard video, image, and control fields. A target video's embedded soundtrack is the audio target; a video with
no audio stream trains as video-only with its audio loss masked. References use `control_directory`, `control_path`, or numbered
`control_path_N` (contiguous from zero — order sets both prompt labels and the rotary timeline):

```json
{"video_path":"/path/to/target.mp4","caption":"A cinematic scene","control_path_0":"/path/to/subject.png","control_path_1":"/path/to/motion.mp4","control_path_2":"/path/to/sound.wav"}
```

For a directory dataset, `control_directory` holds references whose basename matches each target.

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

Ready-made examples for every task:

| Task | Example | Transformer | `--task` | `--h3_training_mode` |
| --- | --- | --- | --- | --- |
| Text-conditioned image | [`image.toml`](../examples/minimax_h3/image.toml) | FL2VA | `t2va` | default |
| Text-to-video+audio | [`t2va.toml`](../examples/minimax_h3/t2va.toml) | FL2VA | `t2va` | default |
| Video-only | [`video_only.toml`](../examples/minimax_h3/video_only.toml) | FL2VA | `t2va` | default |
| Audio-only | [`audio_only.toml`](../examples/minimax_h3/audio_only.toml) | FL2VA | `t2va` | default |
| First-frame I2V | [`i2va.toml`](../examples/minimax_h3/i2va.toml) | FL2VA | `i2va` | default |
| First+last frame | [`fl2va.toml`](../examples/minimax_h3/fl2va.toml) | FL2VA | `fl2va` | default |
| Arbitrary references | [`ref2va.toml`](../examples/minimax_h3/ref2va.toml) | Ref2VA | `ref2va` | `ref2va` |
| Zero-or-more references | [`ref2va_omni.toml`](../examples/minimax_h3/ref2va_omni.toml) | Ref2VA | `ref2va_omni` | `ref2va_omni` |

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

Omit `--audio_vae` for image-only or video-only datasets; omit `--vae` for audio-only. Add `--cache_guidance_empty` if you plan to
use caption dropout or the guidance objective.

`--task` must match how you intend to train: `t2va` (text only), `i2va` (first frame), `fl2va` (first+last), `ref2va`,
`ref2va_omni`. Keyframe tasks take their frames from the target video itself, not from control fields.

```shell
# First-frame I2V
python minimax_h3_cache_text_encoder_outputs.py   --dataset_config dataset.toml   --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors   --task i2va --device cuda

# First+last frame
python minimax_h3_cache_text_encoder_outputs.py   --dataset_config dataset.toml   --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors   --task fl2va --device cuda
```

> [!IMPORTANT]
> Regenerate both caches with the same checkout used for training. An `i2va`/`fl2va` run rejects a cache without keyframe rows
> rather than silently training as `t2va`.

To reduce conditioner VRAM, add `--text_encoder_quantization int8` or `nf4` to the BF16 checkpoint, or load the released
pre-quantized file with `--text_encoder_quantization nvfp4_awq`:

```shell
python minimax_h3_cache_text_encoder_outputs.py   --dataset_config dataset.toml   --text_encoder /models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   --text_encoder_quantization nvfp4_awq   --task t2va --device cuda
```

All three are precision trade-offs, not equivalents. They reduce GPU residency only: the BF16 checkpoint is still memory-mapped
into host address space while weights are converted, so the host RAM requirement is unchanged.

## Training

```shell
accelerate launch minimax_h3_train_network.py \
  --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --dataset_config dataset.toml \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --output_dir output --output_name h3_style \
  --logging_dir logs --log_with tensorboard --log_grad_metrics
```

For Ref2VA, swap the checkpoint and add `--h3_training_mode ref2va` (or `ref2va_omni`).

Adapters target attention and feed-forward projections; norms and timestep/modality calibration stay frozen. LoHa/LoKr are
unsupported. Regional `torch.compile` covers all 50 main blocks and both text-refiner blocks; use `--compile` and optionally
`--compile_auto_cache_size_limit`, `--compile_fallback_to_eager`, or `--inductor_config KEY=VALUE ...`. GPU compilation requires
a working Triton installation; on Windows, install a `triton-windows` build compatible with the installed PyTorch and Python
versions.

Watch progress with `tensorboard --logdir logs`. When training remotely, bind it to a protected interface or reach its loopback
address through an SSH forward rather than exposing it publicly.

### Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `--sdpa`, `--flash_attn`, `--flash3` | `--sdpa` | Attention backend. Each FlashAttention flag needs its package, and `--flash3` needs a Hopper GPU. Both fall back to SDPA on padded batches. |
| `--h3_attn_auto_dispatch` | off | Prefer cuDNN SDPA for large maskless workloads. Changes rounding; benchmark first. |
| `--compile` | off | Regionally compile all H3 blocks with the selected backend/mode. Compatible with full gradient checkpointing and block swap; swapped Linear calls stay eager. |
| `--h3_fused_qk_norm_rope` | off | Use the custom Triton Q/K RMSNorm+RoPE kernel outside compiled graphs. It is faster but changes BF16 rounding, so it is opt-in. |
| `--h3_gradient_checkpointing_cpu_offload_pin_memory` | off | Pin CPU-offloaded checkpoint activations for faster transfers. Requires `--gradient_checkpointing --gradient_checkpointing_cpu_offload` and substantial free system RAM. |
| `--h3_gradient_checkpointing_blocks N` | all 50 | Checkpoint only the last N main blocks. This explicit speed/VRAM trade-off requires `--gradient_checkpointing` and resident eager blocks. |
| `--h3_shift_video` / `--h3_shift_audio` | `12.0` / `3.0` | Per-modality flow shift. Both derive from one shared coordinate, so changing one never desynchronizes the other. |
| `--timestep_sampling` | `uniform` | Use `uniform`, `sigmoid`, or `logsnr`. The dynamic-shift modes double-shift the schedule and ignore H3's temporal extent. |
| `--discrete_flow_shift` | `1.0` | Must stay at the default; H3 applies its own shifts. |
| `--h3_image_flow_shift` | auto | Fixed shift for image batches only. |
| `--h3_video_loss_weight` / `--h3_audio_loss_weight` | `1.0` | Modality weights. `--h3_loss_mode token` switches to element weighting. |

### Memory and speed

Quantize the frozen base, reduce AdaLN, then swap blocks — in that order.

```shell
  --h3_convrot_int8 --h3_convrot_int8_fwd bf16 --h3_adaln_rank 16
```

**This is the recommended configuration.** ConvRot INT8 stores the frozen linear weights at one byte per value,
`--h3_convrot_int8_fwd bf16` evaluates those weights with BF16 activations, and rank 16 replaces the full AdaLN projections with
compact factors.

| Option | Purpose |
| --- | --- |
| `--h3_convrot_int8` | Quantize the released BF16 checkpoint to ConvRot INT8 at load. Rejects `--fp8_base` and `--int8_convrot_base`. |
| `--h3_convrot_int8_fwd bf16` | Recommended. Evaluates the matmul in BF16 without changing stored weights. |
| `--h3_convrot_int8_bwd int8` | INT8 input-gradient path for GPUs without FP8 support. |
| `--fp8_base` | Scaled FP8. `--h3_fp8_quantization_mode` selects `block` (default, fastest), `channel`, or `tensor`. |
| `--int8_convrot_base` | Load the released pre-quantized checkpoint instead of quantizing at load (see below). |
| `--h3_adaln_rank 16` | Reduce the AdaLN projections, the largest parameter group (13.0B of 33.1B), to ~77M. |

To train against the released pre-quantized transformer instead, pass it with `--int8_convrot_base`:

```shell
accelerate launch minimax_h3_train_network.py   --dit /models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors   --int8_convrot_base --sdpa --mixed_precision bf16   --dataset_config dataset.toml   --network_module networks.lora_minimax_h3   --network_dim 16 --network_alpha 16
```

Generic LoRAs passed through `--base_weights` are merged while H3 loads. With a pre-quantized ConvRot base, only affected layers
are dequantized, all requested LoRAs and `--base_weights_multiplier` values are accumulated in FP32, and each layer is
requantized once. The resulting frozen base is then used for ordinary LoRA training.

`--h3_adaln_rank` is more faithful than quantizing those projections and composes with either quantization. It is rejected on
already-pruned and INT8 ConvRot checkpoints.

| Base | Default | With `--h3_adaln_rank 16` |
| --- | --- | --- |
| BF16 | 66.2 GB | 40.5 GB |
| One byte per weight | 33.1 GB | 20.4 GB |

Block swapping streams frozen weights from host memory. It is valid only while the base is frozen:

```shell
  --blocks_to_swap 40 \
  --block_swap_h2d_only \
  --block_swap_ring_size 2
```

`--use_pinned_memory_for_block_swap` can improve transfer bandwidth when the host has enough available memory; leave it disabled
when pinned allocations stall or fail. `--block_swap_granularity layer` streams individual `Linear` layers and supports all 50
blocks, at the cost of more transfers; use the default `block` granularity when it fits. Add
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
  --max_train_epochs 10 --save_every_n_epochs 1 \
  --output_dir output --output_name h3_style
```

This is the balanced minimum-memory preset. If it still exceeds available VRAM, add
`--block_swap_granularity layer`, raise `--blocks_to_swap` to `50`, and use `--block_swap_ring_size 1`; those settings trade
throughput for lower device residency. The BF16 checkpoint itself contains about 61.7 GiB of tensor data, so a full CPU-staging
loader can exceed a 64 GB host after process overhead. H3 block-swap loading instead materializes only CPU-master blocks on the
host. The number of swapped blocks therefore controls both host and device residency.

For the command above, loading the BF16 FL2VA checkpoint peaked at 21.09 GiB process RSS and left 3.02 GiB of model and swap
buffers allocated on the GPU. These are loader figures, not total training requirements: activations, attention workspaces, LoRA
parameters, gradients, and optimizer state are added according to the largest packed batch.

Use `--h3_gradient_checkpointing_cpu_offload_pin_memory` only when the host has substantial free RAM. A 38.9k-row Ref2VA
batch requires about 20–21 GiB of pinned RAM in addition to model-loading and dataset memory.

### Training modes

All of these draw their conditioning from the target itself, so an ordinary `--task t2va` cache is enough.

| Option | Trains |
| --- | --- |
| `--h3_observed_modality {video,audio,random}` | Video-to-audio, audio-to-video, or one adapter covering both plus joint |
| `--h3_extension_video_frames N` / `--h3_extension_audio_latents N` | Continuation from an observed prefix. Counts are in **latent** units and each must be shorter than its target; the two are independent, so setting one leaves the other generated in full |
| `--h3_keyframe_anchors first,11,last` / `--h3_keyframe_random_count N` | Interpolation from arbitrary anchors |
| `--h3_mask_mode {box,border,segment}` | Inpainting, outpainting, temporal infilling |
| `--h3_frame_sigma_jitter 0.2` | Spreads noise level across frames, supervising a range of the schedule per step. Skipped for image batches, which have one frame; `0` disables it |
| `--h3_caption_dropout_rate 0.1` | Trains the unconditional branch; requires `--cache_guidance_empty` |

Extension, keyframes, and masking all claim the observed rows, so **only one may be enabled at a time**.

**Observed modality.** The observed side is pinned to the noise level the release already uses for conditioning of that kind, and
carries no training signal — its loss weight is forced to zero. Datasets must cache both modalities. `random` redraws the task
each step; validation still reports the joint objective so its numbers stay comparable.

**Extension.** `--h3_extension_route` chooses the presentation: `condition_rows` (default) duplicates the context as clean rows,
matching the released keyframe contract but costing sequence length; `per_row_sigma` pins it in place with no extra tokens, at the
cost of intra-block noise levels the released weights have not seen. The observed span is removed from the loss.

**Keyframes.** Entries are `first`, `last`, or a latent frame index. Anchors stay in the loss, matching the released contract.
`last` is the final *pixel* frame, deliberately not the same anchor as the integer `frames - 1`.

**Masking.** Masks are drawn per step, so the occlusion distribution changes without re-caching. `--h3_mask_min_fraction` and
`--h3_mask_max_fraction` bound the masked fraction; `--h3_mask_audio` also hides a run of audio latents. Masks are reduced to the
`(1, 2, 2)` patch grid and a patch counts as generated when any latent inside it is, so at small latent resolutions a wide
fraction range can leave nothing observed.

### Auxiliary objectives

Each adds a full no-gradient forward over the whole packed sequence — roughly `1.25×` a plain step for one, `1.5×` for both. This
matters most for Ref2VA, where the enlarged sequence is re-processed every pass.

| Option | Purpose |
| --- | --- |
| `--h3_guidance_distillation_scale 3` | Guidance-consistent objective using cached empty-text conditioning. `--h3_guidance_loss_form` selects `normalized` or `contrastive`; both share an optimum, but contrastive is `scale²` larger. |
| `--h3_guidance_loss_schedule {sigma,constant}` | `sigma` (default) scales guidance from `1` at the clean endpoint to the configured value at maximum noise, independently for video and audio. `constant` retains the configured scale everywhere. |
| `--h3_base_preservation_loss_weight 0.05` | Penalizes drift from the frozen base's prediction. Anchors to whichever base is loaded, quantized or not. |
| `--crepa` | Temporal representation alignment for video training. |

The H3 base-preservation/distillation-loss technique was proposed by [@Ada123-a](https://github.com/Ada123-a). Ada's community
experiments reported weights around `0.05`–`0.10` with guidance scale `3`. Treat these only as starting points: the appropriate
weight also depends on training length, as well as quantization, rank, dataset, and learning rate. A weight that is useful for a
short run can anchor a long run too strongly to the frozen model and eventually prevent the adapter from learning the dataset.
Monitor training and validation samples, and reduce or disable the weight when preservation begins to dominate; changing it when
resuming training is supported.

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
| `--lora_weight` / `--lora_multiplier` | Attach saved adapters. |
| `--steps` | Sigma grid points including terminal zero, so `20` runs 19 evaluations. |

`--fp8_base`, `--int8_convrot_base`, and the block-swap options are available here too. Output is a synchronized H.264/AAC MP4
with a JSON sidecar recording prompt, geometry, schedule, LoRA names, timings, and memory peaks.

The released weights are CFG-distilled: inference runs one evaluation per step with no negative-prompt branch.

## Dataset configuration reference

See [dataset_config.md](./dataset_config.md) for the shared dataset schema.
