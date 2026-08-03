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

For T2VA training, the native loaders use these four Comfy-Org checkpoint files:

- `diffusion_models/minimax_h3_fl2va_bf16.safetensors`;
- `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`;
- `vae/minimax_h3_video_vae_fp16.safetensors`;
- `vae/minimax_h3_audio_vae_fp32.safetensors`.

The text-cache command also needs the small tokenizer and processor metadata from the official
`FL2VA/text_encoder` directory. The 14 official text-model shards are not needed when the Comfy-Org BF16 text encoder is used. Pass
that metadata directory to `--tokenizer`; Musubi does not import ComfyUI's bundled tokenizer or hard-coded processor classes.

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
so T2VA can train on video-only examples. A present but corrupt audio stream remains an error. Reference image,
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

The default loads Qwen3-VL in BF16. Add `--text_encoder_quantization int8` or
`--text_encoder_quantization nf4` to quantize its Linear weights directly from the same BF16 checkpoint while loading.
INT8 is intended for approximately 32 GB GPUs; NF4 targets 24 GB GPUs. Non-Linear parameters and the cached raw
layer-50 hidden states remain BF16. Both modes are explicit memory/conditioning-precision tradeoffs rather than
numerically equivalent replacements for BF16.

The native loader preserves the released component precision: the Comfy video encoder is FP16 and the audio encoder is FP32.
Normalized cache tensors default to float32, and target video/audio caches use the posterior mean for deterministic reuse. Cache
filenames use Musubi's `mmh3` architecture short name. The latent cache contains normalized video latents, normalized stereo audio
latents as `latents_audio_2x32xT_*` in stereo-major `[2, 32, T]` layout, and explicit loss masks. The conditioning cache is
crop-specific and uses `varlen_mmh3_*` tensors. `--task t2va` stores the raw-caption layer-50 output; `--task fl2va` adds the selected
crop's first and last images, including their vision rows and modality tags. Training accepts only `--task t2va` caches and rejects
FL2VA caches because keyframe VAE conditioning rows are unsupported. Empty-text conditioning is stored only when
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
| `minimax_h3_train_network.py` | `create_training_backend` | The base `fl2va` transformer in text-only T2VA mode; latents and conditioning come from caches |

The adapter uses strict checkpoint loading and preserves the released mixed-precision layout. Training supports Musubi's common
`--fp8_base` and block-swap options. Compilation remains unsupported.

## LoRA training

The H3 trainer follows Musubi's `NetworkTrainer` and LoRA module contracts. The default `fl2va` mode selects the released base
transformer and trains its text-only T2VA path. It packs `[text | target audio | target video]`, jointly noises cached video
and audio, maps the video sigma onto the synchronized audio schedule, and applies a masked joint velocity loss. Attention and
feed-forward projections are adapter targets; norms and timestep/modality calibration stay frozen. First/last-frame latent
conditioning and Ref2VA training are unsupported and rejected rather than silently omitted.

The default `modality` loss mode gives the video and audio means equal weight before applying their explicit loss weights.
`token` is an optional element-weighted reduction over all valid latent values. LoHa/LoKr architecture detection is unsupported.

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
the two-buffer ring overlap transfers with transformer computation. The unpinned path remains available, but uses staged copies
and can be substantially slower for H3's unusually large blocks. H2D-only swapping is valid only while the base transformer is
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

The training backend rejects `--h3_training_mode ref2va`. Reference training requires distinct context and target media; a target
must not be reused as its own reference.

The official model card identifies the released weights as CFG-distilled. H3 inference uses one model evaluation per sampling step,
without a negative-prompt branch. Normal LoRA training therefore uses one conditional
forward and the model's joint flow target. If an authoritative distillation scale is known, `--h3_guidance_distillation_scale` can
enable an experimental two-pass guidance-consistent objective using cached empty-text conditioning. This does not reconstruct an
unconditional model or add negative-prompt inference.
