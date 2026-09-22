# MiniMax Music 3

Diffusion-transformer LoRA training, RVQ encoder distillation, autoregressive
LoRA training, and standalone inference.

## Installation

Install Musubi Tuner with the Music 3 audio dependencies:

```bash
pip install -e ".[minimax_music3]"
```

Text-to-audio inference requires Transformers 5.8.0.

## Model files

Download:

- `Comfy-Org/MiniMax-Music-3/diffusion_models/minimax_music3_dit_fp16.safetensors`
- `Comfy-Org/MiniMax-Music-3/vae/minimax_music3_dav.safetensors`
- `MiniMaxAI/MiniMax-Music3/dav.pth` for latent caching

The AR model, tokenizer, and RVQ depth decoder are loaded from
`MiniMaxAI/MiniMax-Music3`.

## Dataset

Store each audio file with a caption and optional lyrics sidecar:

```text
dataset/
  track.wav
  track.txt
  track.lyrics.txt
```

The caption should describe genre, tempo, key, instrumentation, vocals, and
arrangement. Lyrics may use section tags such as `[verse]`, `[chorus]`, and
`[instrumental]`.

```toml
[[datasets]]
audio_directory = "/data/music"
cache_directory = "/data/music/cache"
batch_size = 1
sample_rate = 44100
max_duration = 30.0
# Optional: group nearby durations, pad within each batch, and mask padding.
duration_bucket_interval = 3.0
```

## Pre-caching

Both caches are required before training:

```bash
python minimax_music3_cache_latents.py \
  --dataset_config music3_dataset.toml \
  --dav /models/dav.pth

python minimax_music3_cache_text_encoder_outputs.py \
  --dataset_config music3_dataset.toml \
  --ar_model MiniMaxAI/MiniMax-Music3
```

To prepare both caches from raw audio and launch training in one resumable
command, use the pipeline wrapper. Additional unknown options are passed to the
DiT trainer:

```bash
python minimax_music3_pipeline.py \
  --dataset_config music3_dataset.toml --dav_encoder /models/dav.pth \
  --ar_model MiniMaxAI/MiniMax-Music3 \
  --dit /models/minimax_music3_dit_fp16.safetensors \
  --vae /models/minimax_music3_dav.safetensors \
  --output_dir output --output_name music3_lora \
  --mixed_precision bf16 --network_dim 64 --max_train_steps 1000
```

The wrapper first writes a MiniMax audio audit report and stops on unreadable,
too-short, missing, or duplicate-cache-name inputs. It then skips existing
caches by default. Use `--no-strict_dataset` to keep diagnostics non-fatal,
`--skip_dataset_report` to omit the audit, `--cache_only` to stop after
preparation, `--no-skip_existing` to rebuild, or `--dry_run` to inspect all
commands without executing them.

## Training

### Diffusion transformer

```bash
accelerate launch --num_cpu_threads_per_process 1 minimax_music3_train_network.py \
  --dataset_config music3_dataset.toml \
  --dit /models/minimax_music3_dit_fp16.safetensors \
  --vae /models/minimax_music3_dav.safetensors \
  --network_module musubi_tuner.networks.lora_minimax_music3 \
  --network_dim 16 --network_alpha 16 \
  --sdpa --mixed_precision fp16 --gradient_checkpointing \
  --optimizer_type AdamW8bit --learning_rate 1e-4 \
  --convrot_int8 --blocks_to_swap 20 \
  --max_train_steps 1000 --save_every_n_steps 100 \
  --output_dir output --output_name music3_lora
```

The trainer uses sigmoid-distributed flow times by default. To run controlled
schedule experiments, select `--music3_timestep_sampling uniform` or
`--music3_timestep_sampling cubic`.

Training batches created with `duration_bucket_interval` are padded to the
longest latent in the batch. Padding is excluded from both self-attention keys
and the flow-matching loss.

Sampling prompts may either provide a precomputed `conditioning_path`, or a
`caption` (or `prompt`), `lyrics`, and optional `duration`. For direct text
conditioning, pass `--ar_model MiniMaxAI/MiniMax-Music3`; the AR model is used
once to build validation conditioning and then released.

Use `--save_state --resume latest` to retain optimizer/scheduler state and
automatically continue from the newest non-empty local Accelerate state.

Memory/regularization controls include `--music3_conditioning_dropout`,
`--music3_input_perturbation`, and sparse segmented activation checkpointing.
For example, `--music3_gradient_checkpointing_interval 2
--music3_gradient_checkpointing_segment_stride 6` checkpoints the first two
blocks in each six-block segment; the defaults checkpoint every block.

MixFlow is available with `--music3_mixflow`; its default
`--music3_mixflow_gamma 0.8` keeps the model time separate from the slowed
latent-interpolation time, matching the Beta(2,1) data-time formulation.

Reference-time and signed-time conditioning are opt-in with
`--music3_flowmap` and `--music3_signed_time`. Reference-time embeddings clone
the pretrained time MLP and use a gated blend; signed-time embeddings start at
zero so enabling them preserves the pretrained model's initial output.

TwinFlow recursive consistency training uses a moving-average LoRA teacher:

```bash
  --music3_flowmap --music3_twinflow --music3_ema_decay 0.999 \
  --music3_twinflow_estimate_order 2 --music3_twinflow_weight 1.0
```

The teacher evaluates progressively cleaner reference times, reconstructs clean
and noise endpoints, and supplies a clamped recursive consistency target. EMA
weights are written to the MiniMax auxiliary checkpoint alongside signed-time
weights.

Self-flow feature alignment is an alternative EMA-teacher objective:

```bash
  --music3_self_flow --music3_ema_decay 0.999 \
  --music3_self_flow_weight 0.5 --music3_self_flow_mask_ratio 0.5
```

It mixes noise times token by token for the student and aligns an intermediate
student representation with a cleaner EMA-teacher representation. The student
and teacher capture layers are configurable with
`--music3_self_flow_student_layer` and `--music3_self_flow_teacher_layer`.
Self-flow and TwinFlow are mutually exclusive.

The trainer automatically restores this companion when resuming a matching
`*-state` directory. It can also be selected explicitly with
`--music3_auxiliary`. Generation discovers the companion next to `--lora`, or
accepts the same explicit option.

`--convrot_int8` quantizes frozen linear weights at load time. LoRA targets the
DiT time embedding, projections, attention, and feed-forward linear layers.

## Inference

```bash
python minimax_music3_generate.py \
  --caption "Acoustic pop, 96 BPM, C major, fingerpicked guitar and soft piano" \
  --lyrics_file lyrics.txt --duration 60 --seed 7 \
  --dit /models/minimax_music3_dit_fp16.safetensors \
  --dav /models/minimax_music3_dav.safetensors \
  --lora output/music3_lora.safetensors \
  --output song.wav
```

Omit `--lora` for base inference. Defaults are AR guidance 1.5, AR top-k 50,
DiT guidance 1.7, and 30 Euler steps.

## RVQ encoder distillation

Build paired DAV-latent/RVQ-code caches from the released generator:

```bash
python minimax_music3_distill_dataset.py \
  --synthetic_items 2048 --duration 8 --variants 3 --steps 30 \
  --dit /models/minimax_music3_dit_fp16.safetensors \
  --dav /models/dav.pth \
  --output_dir rvq_distill --skip_existing
```

Train the audio-to-RVQ encoder:

```bash
python minimax_music3_train_rvq_encoder.py \
  --cache_dir rvq_distill --output rvq_encoder.safetensors \
  --steps 10000 --batch_size 2 --precision bf16
```

Evaluate exact-code and conditioning reconstruction on held-out caches:

```bash
python minimax_music3_eval_rvq_encoder.py \
  --cache_dir rvq_distill --encoder rvq_encoder.safetensors
```

Encode a real dataset. Caption and lyrics sidecars retain the same basename as
the audio files.

```bash
python minimax_music3_encode_rvq.py dataset/*.wav \
  --encoder rvq_encoder.safetensors --dav /models/dav.pth \
  --output_dir dataset/rvq --max_duration 30
```

Train the autoregressive LoRA on the recovered semantic and seven depth code
streams:

```bash
python minimax_music3_train_ar_lora.py \
  --codes_dir dataset/rvq --caption_dir dataset \
  --output_dir output/music3_ar_lora \
  --steps 1000 --frames 64 --continuation_context_frames 128 \
  --caption_dropout_probability 0.1 \
  --rank 16 --alpha 16 \
  --save_every 250 --checkpoints_total_limit 3 \
  --resume_from_checkpoint latest
```

AR checkpoints contain LoRA weights, optimizer and scheduler state, and Python,
CPU, and CUDA random-number-generator states. `latest` resumes from the highest
complete `step-*` checkpoint and checkpoint retention removes older step
directories after a successful save.

For corpus-preserving regularization, provide a second RVQ corpus. Regularized
batches match the frozen base planner's top-k semantic and depth distributions
instead of learning their ground-truth codes:

```bash
  --regularization_codes_dir /data/base-rvq \
  --regularization_caption_dir /data/base-captions \
  --regularization_probability 0.5 --regularization_top_k 64
```

Explorative route training and NextLat can be enabled together. XM adds a
trainable embedding per candidate route, evaluates every route, and backpropagates
only the lowest block-balanced candidate loss. NextLat predicts the next captured
hidden state with a zero-initialized residual MLP:

```bash
  --xm_enabled --xm_candidate_count 2 --xm_block_size 16 \
  --nextlat_enabled --nextlat_block_index -1 \
  --nextlat_weight 0.1 --nextlat_state_loss smooth_l1
```

Their trainable auxiliary tensors are stored in
`training_auxiliary.safetensors` and in resumable step state.

Use the AR LoRA for standalone generation:

```bash
python minimax_music3_generate.py \
  --caption "Atmospheric drum and bass, rolling breaks, deep bass and pads" \
  --lyrics "[instrumental]" --duration 60 --seed 7 \
  --dit /models/minimax_music3_dit_fp16.safetensors \
  --dav /models/minimax_music3_dav.safetensors \
  --ar_lora output/music3_ar_lora --output song.wav
```

## Limitations

MiniMax has not released the audio-to-RVQ encoder. Current training adapts only
the DiT renderer unless an RVQ encoder is distilled and used to cache target
codes. `dav.pth` provides continuous DiT latents, not RVQ codes.

The distilled encoder approximates the private tokenizer. Exact code recovery
is incomplete, and autoregressive fine-tuning is sensitive to encoder error,
teacher-forcing exposure bias, corpus coverage, and sequence length. It can
change learned musical structure, but it is not equivalent to training with
the original MiniMax encoder.
