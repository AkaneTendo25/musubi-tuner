# YuE2

## Overview / 概要

Musubi Tuner supports LoRA training and inference for **YuE2** ([m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)), a 3B lyrics-to-song model. YuE2 has two transformer stacks that share one context:

- **AR** (autoregressive): a Qwen-style causal LM that writes an optional ABC score and then one semantic codec token per latent frame (25 frames per second).
- **NAR** (non-autoregressive): a flow-matching stack that turns the codec tokens into 64-channel VAE latents. It attends to the AR K/V of the prompt and the codes. The YuE2 VAE decodes the latents to 48 kHz stereo audio.

One trainer covers the three training modes (`--train_branches nar`, `ar`, `ar,nar`). It supports bf16, ConvRot int8, scaled fp8, gradient checkpointing with activation CPU offload, block swap on both stacks, `torch.compile`, and the SDPA/flash/xformers backends. It can compute validation losses, keep a best checkpoint, render audio previews and export LoRAs in the ComfyUI and other YuE2 LoRA formats.

This feature is experimental. No LoRA trained with it has yet shown a clear audible improvement over the base model. See [Known limitations](#known-limitations--既知の制限).

Use `--help` for pre-caching, training and inference options. Common musubi options are described in the [HunyuanVideo documentation](./hunyuan_video.md) and the [advanced configuration](./advanced_config.md).

<details>
<summary>日本語</summary>

Musubi Tuner は、歌詞から楽曲を生成する 3B モデル **YuE2**（[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)）の LoRA 学習と推論に対応しています。YuE2 は一つのコンテキストを共有する 2 つのスタックで構成されます。

- **AR**（自己回帰）: 任意の ABC 譜面を書き、その後 latent フレーム（毎秒 25 フレーム）ごとに 1 つのセマンティックコーデックトークンを生成する Qwen 系の causal LM。
- **NAR**（非自己回帰）: コーデックトークンを 64 チャンネルの VAE latent に変換する flow matching スタック。プロンプトとコードの AR K/V を参照します。YuE2 VAE が latent を 48 kHz ステレオ音声にデコードします。

一つのトレーナーで `--train_branches nar` / `ar` / `ar,nar` の 3 つの学習モードを扱います。bf16、ConvRot int8、scaled fp8、gradient checkpointing（アクティベーションの CPU オフロード付き）、両スタックの block swap、`torch.compile`、SDPA/flash/xformers に対応します。検証ロス、ベストチェックポイント、学習中の音声サンプル、ComfyUI などへの LoRA エクスポートも利用できます。

この機能は実験的なものです。この機能で学習した LoRA で、ベースモデルに対する明確な聴感上の改善はまだ示されていません（[既知の制限](#known-limitations--既知の制限)を参照）。

</details>

## Licensing / ライセンス

Musubi Tuner's code is Apache-2.0, but the models used by this workflow have their own licences. The chain is non-commercial:

| Component | Licence | Notes |
|---|---|---|
| YuE2-3B weights (HF and the Comfy-Org repackages) | CC BY-NC 4.0 | Every LoRA trained on them is a derivative. |
| YuE2-Vae weights | CC BY-NC 4.0 | The VAE *code* (Oobleck, SnakeBeta) is MIT. See `tests/yue2_ref/licenses/`. |
| Mothersuperior semantic tokenizer heads and `nar_lora_joint_*` adapters | CC BY-NC 4.0 | From the model card front matter. |
| SheetSage2 (ABC transcription; not shipped here) | CC BY-NC 4.0 | Relevant only if you use it to make `.abc.txt` files. |
| MERT-v2-FullSong (feature extractor for the semantic codes) | CC BY-NC 4.0 | Front matter of the model card in the downloaded `m-a-p/MERT-v2-FullSong` snapshot. |

LoRAs made with this workflow inherit CC BY-NC 4.0 restrictions. The training audio needs its own rights. Parts of the YuE2 modules are derived from third-party code (the official YuE2 inference code, stable-audio-tools, BigVGAN and torchaudio); [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) lists each item and its licence notice. The official `yue2_infer` model code vendored under `tests/yue2_ref/` is used only by tests and includes the licence files of its wheel; see `tests/yue2_ref/NOTICE`.

<details>
<summary>日本語</summary>

Musubi Tuner のコードは Apache-2.0 ですが、使用するモデルには個別のライセンスがあります。YuE2-3B の重み、YuE2-Vae の重み、Mothersuperior のトークナイザーヘッドと `nar_lora_joint_*` アダプタ、SheetSage2、MERT-v2-FullSong はいずれも CC BY-NC 4.0（非営利）です。これらで学習した LoRA も非営利の制約を受けます。学習用音声の権利は別途必要です。YuE2 モジュールの一部はサードパーティのコードに由来します。各項目とライセンス表記は [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を参照してください。

</details>

## Download the model / モデルのダウンロード

Pass any of these checkpoints as `--dit`; the keys determine the format:

- **Hugging Face** [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B): `model.safetensors` (bf16) and `qwen.tiktoken`. Pass the separate [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) directory with `config.json` and weights as `--vae`. The `qwen.tiktoken` tokenizer needs `pip install tiktoken`; alternatively, pass a tokenizer `.json` with `--tokenizer`.
- **ComfyUI bf16** [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) `checkpoints/yue2_3b_bf16.safetensors`: an all-in-one file with the model, the VAE (fp32) and the tokenizer JSON. `--vae` and `--tokenizer` may be omitted. After loading, the model tensors are bit-identical to the HF file.
- **ComfyUI int8** `checkpoints/yue2_3b_int8_convrot.safetensors`: pre-quantized ConvRot int8. It is detected automatically and trains with a frozen int8 base. Its VAE is stored as fp16, so it is not used for latent caching (see `--allow_fp16_vae`), and `--base_weights` cannot be merged into it.

For the AR targets and default NAR context, also download:

- [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong): the audio feature extractor. Pass its hub id or a local directory to `--mert_model`. Its remote code imports torchaudio, but torchaudio is not required; a pure-torch stand-in is installed while MERT loads.
- A Mothersuperior semantic tokenizer head from [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4): `tokenizer_head_joint_v4.safetensors`, `tokenizer_head_joint_v9.safetensors`, `tokenizer_head_v5_30k.safetensors` (`.safetensors` or `.pt`), plus the matching companion NAR adapters `nar_lora_joint_v{4,5,8,9}*.safetensors`.

## Dataset configuration / データセット設定

YuE2 uses audio datasets (`audio_directory` or `audio_jsonl_file`). See [Audio Datasets (YuE2)](./dataset_config.md#audio-datasets-yue2) for all keys. Minimal config:

```toml
[general]
batch_size = 1

[[datasets]]
audio_directory = "/data/artist"        # song.flac + song.caption.txt (style) + song.lyrics.txt (+ song.abc.txt)
cache_directory = "/data/artist/cache_yue2"
trigger = "sv_artist"                   # optional, prepended to the style
segment_extraction = "full"             # full | head | chunk | slide
max_seconds = 360.0
validation_split = 0.1                  # whole songs held out for validation losses and reconstruct previews
```

JSONL records: `{"audio_path": "a.flac", "style": "dark synthwave, female vocals", "lyrics": "[verse]\n...", "abc": "X:1\n...", "abc_mode": "full", "song_id": "a", "start": 12.0, "end": 72.0}`.

- Style is the `[Tags]` text of the YuE2 protocol, capped at 1500 characters. Songs without lyrics use `--instrumental_lyrics` (default `[instrumental]`).
- An ABC score is optional. Its flavour (`melody` or `full`, set per record or by a first line `%%yue2_abc_mode melody|full`) selects the chain-of-thought instruction. No ABC transcription tool is shipped with this version; scores must come from your own sidecars or JSONL fields.
- `batch_size = 1` is recommended. With block swap on a trained stack, `batch_size > 1` is rejected (one backward per batch; see [Memory](#memory--メモリ)).
- Only segments that start at the beginning of a song train the AR branch: `full` and `head` do; later `chunk`/`slide` windows train only the NAR. The end-of-music token is a target only when the segment reaches the true song end.

## Pre-caching / 事前キャッシング

### Latents and semantic codes / latent とセマンティックコード

```bash
python yue2_cache_latents.py --dataset_config dataset.toml \
  --vae /models/m-a-p__YuE2-Vae \
  --mert_model m-a-p/MERT-v2-FullSong \
  --semantic_head /models/ms/tokenizer_head_joint_v9.safetensors
```

- Latents come from the VAE posterior mean (fp32 VAE). Encoding uses 750-frame chunks with 50 frames of context (`--vae_chunk_frames`, `--vae_overlap_frames`); chunked encoding matches a full encode to rel. L2 1.4e-4. Segments are sliced from one encode of the whole record.
- One of `--semantic_head` or `--no_codes` is required. With a head, MERT-v2 layer-20 features go through the head and `codes_int64` (one code per latent frame) is stored next to the latents. `--no_codes` caches latents only; that supports just `--nar_context text_only` and no AR training.
- Without `--vae`, the VAE is read from a ComfyUI `--dit`. The int8 all-in-one stores the VAE as fp16 and is refused unless `--allow_fp16_vae`.
- `--sequential_models` runs the VAE and MERT in two passes (lower peak VRAM, same tensors). `--latent_dtype bfloat16` halves the cache size. `--mert_revision` and `--mert_dtype` (bf16 autocast by default) are also available.
- `--skip_existing` checks fingerprints for the VAE, chunking, audio file, record extent, head sha256 and MERT identity. If only the head changed, the latents are reused and only the codes are recomputed.
- Measured on an H100: about 0.5 s per minute of audio. Peak VRAM 5.3 GiB single pass, 3.6 GiB with `--sequential_models`.

### Text / テキスト

```bash
python yue2_cache_text_encoder_outputs.py --dataset_config dataset.toml --dit /models/yue2_3b_bf16.safetensors
```

YuE2 has no text encoder. The "text cache" stores the protocol token ids of every record: the prompt for `cot` off, melody and full, the matching negative prefixes, and the ABC ids. Tokenizer lookup order is `--tokenizer`, the ComfyUI `--dit`, then `qwen.tiktoken` next to an HF `model.safetensors`. `--instrumental_lyrics` is written into the cache and checked by the trainer. `--debug_mode console` prints the prompts and writes nothing.

### Semantic tokenizer dialect (read this before AR training) / セマンティックトークナイザーの「方言」

The official audio-to-semantic tokenizer of YuE2 is not released. Codes from real audio come from the published MERT-v2 + Mothersuperior tokenizer head. That head agrees with YuE2's own codes only about 18-23% of the time (top-1, measured here on songs generated by YuE2: joint_v4 0.18, joint_v9 0.20-0.23, v5_30k 0.20-0.23). Its codes are a "dialect" of YuE2's token language:

- An AR LoRA learns to emit head-dialect tokens. At inference, pair it with the matching companion NAR adapter (`nar_lora_joint_vN` of the same head version), or with a NAR LoRA trained on the same head's codes.
- Heads disagree with each other (v4 vs v9 agree on about 31% of codes). Cache the whole dataset with one head; the trainer refuses caches that mix heads and records the head sha256 in `ss_yue2_semantic_head`.
- For NAR training with `--nar_context codes` (the default), fold the companion adapter of the same head into the base with `--base_weights` so the NAR starts from a model that already reads the dialect (see the joint recipe).

## Training / 学習

Use `yue2_train_network.py` for training. It forces bf16 model weights (`--mixed_precision bf16` is required; fp16 is rejected) and the fp32 VAE. `--network_module networks.lora_yue2` is the default.

### Recipes / レシピ

**NAR style LoRA** (timbre, mix, production; the AR is untouched):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 yue2_train_network.py \
  --dit /models/yue2_3b_bf16.safetensors --vae /models/m-a-p__YuE2-Vae \
  --dataset_config dataset.toml --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --train_branches nar --network_dim 32 --network_alpha 32 \
  --optimizer_type adamw --learning_rate 1e-4 --max_grad_norm 1.0 --lr_warmup_steps 50 \
  --max_train_steps 1600 --save_every_n_steps 200 \
  --validate_every_n_steps 100 --save_best_validation \
  --sample_prompts prompts.json --sample_every_n_steps 200 \
  --output_dir output --output_name my_style --yue2_export_formats comfy
```

**Artist AR + NAR joint LoRA** (melody, phrasing and sound; needs cached codes):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 yue2_train_network.py \
  --dit /models/yue2_3b_bf16.safetensors --vae /models/m-a-p__YuE2-Vae \
  --dataset_config dataset.toml --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --train_branches ar,nar --network_dim 32 --network_alpha 32 \
  --network_args "ar_dim=16" "ar_alpha=16" \
  --base_weights /models/ms/nar_lora_joint_v9_comfyui.safetensors \
  --ar_kl_weight 0.2 --ar_lr_ratio 0.5 --abc_dropout 0.5 --caption_dropout 0.1 \
  --optimizer_type adamw --learning_rate 1e-4 --max_grad_norm 1.0 --lr_warmup_steps 50 \
  --max_train_steps 1200 --save_every_n_steps 200 --validate_every_n_steps 100 --save_best_validation \
  --sample_prompts prompts.json --sample_every_n_steps 200 --sample_ar_resident \
  --output_dir output --output_name my_artist \
  --yue2_export_formats comfy --yue2_export_include_base_weights
```

- `--base_weights` merges the companion NAR adapter into the bf16 weights while loading (any YuE2 LoRA format works). The trained LoRA is a delta on top of it. `--yue2_export_include_base_weights` rank-concatenates the companion into the exported files to make them self-contained. The plain `{output_name}.safetensors` is always the delta only.
- The head used for caching and the companion must be the same version (here v9).
- Memorisation can set in early (it has been reported after about 1,500 steps on small sets), and AR CE can fall to near zero within a few hundred steps. Choose the checkpoint by validation loss and listening. Keep the KL term, and consider a lower AR learning rate (`--ar_lr_ratio`) and rank.

**Text-only fallback** (no MERT/head; codes are not cached):

```bash
python yue2_cache_latents.py --dataset_config dataset.toml --vae /models/m-a-p__YuE2-Vae --no_codes
python yue2_train_network.py ... --train_branches nar --nar_context text_only
```

This is off-protocol: the NAR sees only the text prefix during training but real codes at inference. LoRAs trained this way have been reported to work best at strength 2.0 and not to clone voices. Use it only when codes cannot be produced. `--nar_text_only_rope full` (default) keeps the inference positions (as if the codes were present); `compact` places the NAR right after the prefix.

### YuE2 options / YuE2 固有のオプション

Mode and conditioning:

| Option | Default | Meaning |
|---|---|---|
| `--train_branches` | `nar` | `nar`, `ar` or `ar,nar` |
| `--nar_context` | `codes` | `codes`: prefix + window codes + end-of-music, as in inference. `text_only`: prefix only |
| `--nar_codec_dropout` | 0.0 | probability of hiding the codes from the NAR (codes mode) |
| `--nar_text_only_rope` | `full` | NAR positions for `text_only`: `full` (inference layout) or `compact` |
| `--nar_window_frames` | 1500 | NAR training window in frames (60 s); 0 = whole segment; capped to the 24,576-token context |
| `--cot` | `auto` | ABC usage: `auto` (the record's flavour when it has ABC, else off), `off`, `melody`, `full` |
| `--abc_dropout` | 0.5 | probability of training a scored record without its score |
| `--caption_dropout` | 0.0 | probability of the protocol negative prefix for the AR branch (CFG training; the NAR context stays positive) |
| `--nar_caption_dropout` | 0.0 | negative prefix for the NAR context (experiments) |
| `--instrumental_lyrics` | `[instrumental]` | lyrics of songs without lyrics; must match the text cache |

AR loss:

| Option | Default | Meaning |
|---|---|---|
| `--ar_ce_targets` | `codec` | `codec` or `codec_abc` (also the ABC score tokens) |
| `--ar_max_tokens` | 0 | AR codes per item from the song start (0 = whole song). Set it for songs near the 24,576-token context |
| `--ar_ce_chunk` | 512 | rows per checkpointed CE/KL chunk (vocab-sized logits never exist for the whole song) |
| `--ar_pad_multiple` | 256 with `--compile`, else 0 | right-pad AR passes to a multiple (exact under causal attention; fewer recompiles) |
| `--ar_kl_weight` | 0.2 | KL(base ‖ LoRA) on the AR logits |
| `--ar_ce_weight` / `--flow_loss_weight` | 1.0 / 1.0 | loss weights |
| `--ar_lr_ratio` | 1.0 | AR LoRA learning rate = `--learning_rate` × ratio |
| `--ar_minted_pack` / `--ar_replay_fraction` / `--ar_minted_val_fraction` | none / 0.0 / 0.1 | replay YuE2-generated ("minted") songs in the AR loss (ComfyUI-FL-YuE2 `.pt` kit or `.json` manifest, or JSONL with codes); logs `val/minted_ce` |

NAR flow and I/O:

| Option | Default | Meaning |
|---|---|---|
| `--timestep_sampling` | `sigmoid` | `uniform`, `sigmoid`, `shift`, `logsnr` or `beta` (`--beta_timestep_ab a,b`, default `2.0,2.0`, clipped to `--min/max_timestep` or [20, 980]) |
| `--first_timestep_chance` | 0.0 | probability of forcing t = 1 (pure noise) |
| `--t_embed_dtype` | `bf16` | time-embedding input as the official inference code computes it (`bf16`) or plain fp32 |
| `--train_io` | `none` | NAR `vae2llm`/`llm2vae`: `none`, `lora`, or `full` (a trainable full-rank diff) |
| `--io_lr` | = lr | learning rate of the I/O modules |

`--weighting_scheme` must be `none`; `--discrete_flow_shift` defaults to 1.0; `--dim_from_weights` is not supported (use `--network_weights`).

Validation and samples:

| Option | Default | Meaning |
|---|---|---|
| `--validate_every_n_steps` | none | validation losses on the held-out songs (`validation_split` / `is_validation` datasets) |
| `--validation_timesteps` | `0.2,0.5,0.8` | flow times of `val/flow` |
| `--validation_noise_seed` | 1234 | fixed validation noise |
| `--validation_max_items` | 8 | validation segments per run |
| `--save_best_validation` | off | write `{output_name}-best.safetensors` when `val/total` improves |
| `--sample_mode` / `--sample_seconds` | `reconstruct` / 30 | defaults for sample prompts |
| `--sample_ode_state_dtype` | `bf16` | ODE state dtype of previews |
| `--sample_ar_resident` | off | keep the whole AR stack on the GPU while previews decode tokens (with block swap) |
| `--yue2_export_formats` | none | also write each saved LoRA as `comfy`, `hf`, `fl` |
| `--yue2_export_include_base_weights` | off | rank-concatenate `--base_weights` (at their `--base_weights_multiplier`) into those exports |

Memory and precision (shared with the generator):

| Option | Default | Meaning |
|---|---|---|
| `--blocks_to_swap` | 0 | blocks per stack to swap to CPU (0-26) |
| `--ar_blocks_to_swap` / `--nar_blocks_to_swap` | = `--blocks_to_swap` | per-stack counts |
| `--convrot_int8` / `--convrot_int8_bwd` | off / `bf16` | ConvRot int8 base (`int8` routes the backward through the int8 GEMM, needs triton) |
| `--fp8_base --fp8_scaled` | off | scaled fp8 base (both flags are required) |
| `--quantize_lm_head` | off | also quantize the AR `lm_head` (default: bf16 for exact CE) |
| `--prequant_lm_head` | `auto` | `lm_head` of the pre-quantized int8 file: dequantized when AR logits are needed (`auto`), `keep` or `dequant` |
| `--sdpa_gqa` | `repeat` | SDPA grouped-query attention: expand K/V heads (`repeat`) or `native` |
| `--attn_query_tile` | 256 | query tile for `--split_attn` (exact causal masks per tile) |
| `--tokenizer` / `--vae` | from `--dit` | as for caching |

Attention: `--sdpa` (recommended), `--flash_attn`, `--flash3`, `--xformers`, optionally with `--split_attn`. `--sage_attn` is rejected for training (no backward).

### Network arguments / ネットワーク引数

`--network_args key=value ...`:

| Key | Default | Meaning |
|---|---|---|
| `lora_layout` | `split` | `split`: separate q/k/v and gate/up LoRAs inside the fused Linears (one LoRA per projection of the HF checkpoint). `fused`: one LoRA per fused Linear |
| `targets` | `attn,mlp` | block targets |
| `ar_dim`, `ar_alpha`, `nar_dim`, `nar_alpha` | `--network_dim/alpha` | per-branch rank and alpha |
| `io_modules` | `vae2llm,llm2vae` | I/O modules for `--train_io lora/full` |
| `include_time_embedder` | False | also train the time embedder MLP (with `--train_io`) |
| `io_dim`, `io_alpha` | nar values | rank/alpha of I/O LoRAs |

The generic musubi keys also work: `exclude_patterns`, `include_patterns`, `rank_dropout`, `module_dropout`, `loraplus_lr_ratio`, `verbose`. Unknown keys raise an error. The trainer fills in `branches`, `train_io`, `ar_lr_ratio` and `io_lr` from its flags and records them in `ss_network_args`.

`--network_weights` and `--base_weights` accept every known YuE2 LoRA format: musubi native, ComfyUI / ai-toolkit, yue2-lora-v1 (Starnodes, YuE2 Studio), fl-yue2-lora-v1 (ComfyUI-FL-YuE2), and Mothersuperior `.safetensors` / `.pt` (full I/O weights are turned into diffs against the checkpoint).

### Memory / メモリ

Measured on an H100 with the ComfyUI bf16 checkpoint: songs of 183-234 s (the AR runs on the whole song, about 5,000-6,300 tokens), 60 s NAR window, bs 1, rank 16, `--gradient_checkpointing`, `--sdpa`. `peak` is `torch.cuda.max_memory_allocated` of the whole run. The PyTorch allocator reserves roughly 1-2 GB more than that. `s/it` is the mean over the run, including warm-up: 60 steps for the precision rows, 20-40 steps for the rows marked \*.

| Mode | Settings | peak GB | s/it |
|---|---|---|---|
| NAR | bf16 | 6.8 | 0.32 |
| NAR | `--fp8_base --fp8_scaled` | 4.3 | 0.32 |
| NAR | `--convrot_int8` | 4.2 | 0.38 |
| NAR | pre-quantized int8 file | 3.9 | 0.40 |
| NAR | bf16 + validation + previews \* | 8.6 | 0.30 |
| NAR | `--compile` \* | 6.8 | 0.16 |
| AR | bf16 | 7.2 | 0.56 |
| AR | `--fp8_base --fp8_scaled` | 6.0 | 0.58 |
| AR | `--convrot_int8` | 5.9 | 0.59 |
| AR | pre-quantized int8 file | 5.6 | 0.59 |
| joint | bf16 | 10.4 | 0.84 |
| joint | `--fp8_base --fp8_scaled` | 7.8 | 0.86 |
| joint | `--convrot_int8` | 7.7 | 0.93 |
| joint | pre-quantized int8 file | 7.4 | 0.98 |
| joint | `--blocks_to_swap 14` \* | 8.1 | 1.68 |
| joint | `--blocks_to_swap 14 --block_swap_h2d_only` \* | 8.5 | 0.87 |
| joint | `--gradient_checkpointing_cpu_offload` \* | 10.1 | 1.38 |
| joint | `--compile` \* | 10.4 | 0.52 |

The AR branch dominates, and its activations grow with song length. Songs longer than about 4 minutes were not measured. `--ar_max_tokens` bounds the AR length (codes from the song start).

Notes on the table:

- On the H100, quantization lowers memory, not step time. `--fp8_base --fp8_scaled` ran at the bf16 step time (+2-4%). The int8 bases were 2-25% slower than fp8, mostly in NAR training: the int8 layers need a dequantized weight and an inverse rotation in the backward pass.
- The int8 GEMM kernel is autotuned once per new sequence length (about 0.75 s per layer shape). Songs of different lengths each add this once, so short runs show a higher mean step time than long ones.
- `--blocks_to_swap 14` lowers the joint peak by about 2.6 GB. In the same short runs, the step time was 0.87 s with `--block_swap_h2d_only`, 0.78 s without swap and 1.68 s with the default swap.
- `--gradient_checkpointing_cpu_offload` saved 0.6 GB at these song lengths and made the step 1.8x slower (short runs). The saving should grow with the sequence length; this was not measured.
- `--compile` lowered the step time (short runs: NAR 0.30 → 0.16 s, joint 0.78 → 0.52 s) and did not change the peak.
- Only the H100 was measured. Step times on other GPUs will differ; the peak memory figures do not depend on the GPU model.
- `--nar_window_frames 0` (whole-song NAR window) was not measured.

For a quantized base, prefer `--convrot_int8` (or the pre-quantized ComfyUI int8 file, which uses the same format) to `--fp8_base --fp8_scaled`. It is closer to the bf16 model. The table shows forward relative error against bf16 on the real weights and the first-step NAR loss of the training matrix:

| Base | AR logits | NAR velocity | NAR loss at step 1 |
|---|---|---|---|
| bf16 | 0 | 0 | 0.9384 |
| `--convrot_int8` | 0.0075 | 0.054 | 0.9385 |
| pre-quantized int8 file | 0.0079 | 0.051 | 0.9388 |
| `--fp8_base --fp8_scaled` | 0.0108 | 0.068 | 0.9395 |

`--convrot_int8` costs step time. On the H100, ConvRot int8 was 2-25% slower than fp8 (see the table). `--convrot_int8_bwd int8` did not make the backward faster on the H100 for these layer sizes.

**24 GB** (e.g. RTX 3090/4090): joint bf16 as in the recipe above, no swap. Songs of 5+ minutes were not measured; if they do not fit, add `--gradient_checkpointing_cpu_offload`, or `--ar_max_tokens 6000`.

**16 GB** (e.g. RTX 4060 Ti 16GB, 4080): joint with a quantized base, or bf16 with swap:

```bash
# quantized base (bf16 LoRA on an int8 base)
--train_branches ar,nar --convrot_int8 --gradient_checkpointing
# or bf16 base with block swap (H2D-only is much faster than the default swap)
--train_branches ar,nar --blocks_to_swap 14 --block_swap_h2d_only --use_pinned_memory_for_block_swap --gradient_checkpointing
```

NAR-only style training fits in 8 GB with a quantized base. With block swap on a trained stack, `batch_size` must be 1 (each batch must be a single backward pass). The embedding and `lm_head` stay on the CPU in NAR-only training, and the NAR stack stays on the CPU in AR-only training. Both are moved to the GPU only for previews.

<details>
<summary>日本語</summary>

H100 と ComfyUI の bf16 チェックポイントでの計測値です。曲の長さは 183～234 秒です。AR は曲全体（約 5,000～6,300 トークン）で学習します。そのほかの条件は NAR ウィンドウ 60 秒、bs 1、rank 16、`--gradient_checkpointing`、`--sdpa` です。`peak` は実行全体の `torch.cuda.max_memory_allocated` で、PyTorch のアロケータはこれより 1～2 GB ほど多く確保します。数値は英語部分の表を参照してください。

AR ブランチがメモリの大半を占め、そのアクティベーションは曲の長さに比例して増えます。約 4 分を超える曲は計測していません。`--ar_max_tokens` で AR の長さ（曲頭からのコード数）を制限できます。

表についての補足：

- H100 では、量子化で下がるのはメモリであり、ステップ時間ではありません。`--fp8_base --fp8_scaled` のステップ時間は bf16 とほぼ同じ（+2～4%）でした。int8 の base は fp8 より 2～25% 遅く、差は主に NAR の学習で出ます。int8 の層は backward で重みの逆量子化と逆回転が必要なためです。
- int8 GEMM カーネルは新しい系列長ごとに 1 回オートチューニングされます（層の形状 1 つにつき約 0.75 秒）。長さの異なる曲ごとに 1 回ずつ加わるので、短い実行ほど平均ステップ時間が長く見えます。
- `--blocks_to_swap 14` で joint のピークは約 2.6 GB 下がります。同じ短い実行でのステップ時間は、`--block_swap_h2d_only` 付きで 0.87 秒、スワップなしで 0.78 秒、デフォルトのスワップで 1.68 秒でした。
- `--gradient_checkpointing_cpu_offload` はこの曲の長さでは 0.6 GB の削減にとどまり、ステップは 1.8 倍遅くなりました（短い実行）。系列長に応じて削減量は増えるはずですが、計測していません。
- `--compile` はステップ時間を短縮し（短い実行: NAR 0.30 → 0.16 秒、joint 0.78 → 0.52 秒）、ピークは変わりませんでした。
- 計測したのは H100 だけです。ほかの GPU ではステップ時間が異なりますが、ピークメモリの値は GPU の機種に依存しません。
- `--nar_window_frames 0`（曲全体を NAR ウィンドウにする）は計測していません。

量子化した base で学習する場合は、`--fp8_base --fp8_scaled` より `--convrot_int8`（または同じ形式の ComfyUI 事前量子化 int8 ファイル）を推奨します。bf16 モデルとの差が小さいためです。実重みで計測した bf16 に対する forward の相対誤差（AR logits / NAR velocity）と、学習マトリクスの 1 ステップ目の NAR loss は次のとおりです：`--convrot_int8` 0.0075 / 0.054 / 0.9385、事前量子化 int8 ファイル 0.0079 / 0.051 / 0.9388、`--fp8_base --fp8_scaled` 0.0108 / 0.068 / 0.9395（bf16 の loss は 0.9384）。ステップ時間とのトレードオフがあり、H100 では ConvRot int8 は fp8 より 2～25% 遅くなりました（表を参照）。`--convrot_int8_bwd int8` は、H100 のこの層サイズでは backward を速くしませんでした。

**24 GB**（RTX 3090/4090 など）：上のレシピどおり joint を bf16 で、スワップなしで学習できます。5 分以上の曲は計測していません。収まらない場合は `--gradient_checkpointing_cpu_offload` か `--ar_max_tokens 6000` を追加してください。

**16 GB**（RTX 4060 Ti 16GB、4080 など）：量子化した base で joint を学習するか、bf16 でブロックスワップを使います（英語部分のコマンド例を参照）。

NAR のみのスタイル学習は、量子化した base なら 8 GB に収まります。学習対象のスタックでブロックスワップを使う場合、`batch_size` は 1 にする必要があります（1 バッチを 1 回の backward にするため）。NAR のみの学習では埋め込みと `lm_head` は CPU に置かれ、AR のみの学習では NAR スタックが CPU に置かれます。どちらもプレビューのときだけ GPU に移されます。

</details>

### Sample generation during training / 学習中のサンプル生成

Use JSON or TOML (a list of objects) for prompts. Keys: `prompt` (style), `lyrics`, `lyrics_file`, `cot`, `abc`, `abc_file`, `seconds`, `seed`, `sample_steps` (ODE steps, default 32), `cfg_scale`, `mode` (`render` or `reconstruct`), `reconstruct_cache`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `abc_temperature`, `abc_top_p`, `abc_top_k`. Relative file paths are resolved from the prompt file's directory.

```json
[
  {"mode": "reconstruct", "seconds": 20, "seed": 7},
  {"prompt": "sv_artist, dark synthwave, female vocals", "lyrics_file": "verse.txt", "mode": "render", "cot": "off", "seconds": 30, "seed": 3}
]
```

- `reconstruct` (default): the NAR renders the cached codes of a held-out song (validation item `enum % n`, or `reconstruct_cache`). This is fast and shows what the NAR learned. The ground-truth decode is written once as `*_gt.flac`.
- `render`: the full pipeline (AR tokens, NAR, VAE). It is slow under block swap: an 8 s preview took about 150 s with `--blocks_to_swap 14`, and about 36 s with `--sample_ar_resident`.
- Line form: `style text --lyf lyrics.txt --cot off --abcf score.abc --sec 30 --d 42 --mode render --recon cache_file.safetensors`.

Previews are written as 48 kHz FLAC under `output_dir/sample`.

### Validation and the best checkpoint / 検証とベストチェックポイント

With a held-out split and `--validate_every_n_steps N`, the trainer logs `val/flow` (mean over `--validation_timesteps`, fixed noise, centre window), `val/flow@t`, `val/ar_ce`, `val/ar_kl`, `val/total` (loss-weighted) and, with a minted pack, `val/minted_ce`. During validation, LoRA dropouts are off and schedule-free optimizers are switched to eval. Two calls on the same weights give the same numbers. `--save_best_validation` keeps `{output_name}-best.safetensors` with full metadata and the run's session id. The best value is saved with `--save_state` and restored by `--resume`. When the minted pack is the only validation data, `val/total` is `val/minted_ce`. Latent-space loss does not always track what listeners hear; also compare renders.

### Metadata / メタデータ

Saved LoRAs carry `modelspec.*` (architecture `YuE2-3B`, resolution `48000x2`), `ss_network_args`, and `ss_yue2_*` keys. These include the protocol version, branches, layout, NAR context and window, cot and dropouts, loss weights, timestep sampling, tokenizer fingerprint, semantic head sha256 and MERT identity, base weights (name and sha256), base quantization, checkpoint layout, cache versions and the instrumental lyrics.

## Exports and conversion / エクスポートと変換

`yue2_convert_lora.py` converts between the formats:

```bash
python yue2_convert_lora.py --input my_artist.safetensors --output my_artist.comfy.safetensors --to comfy --verify
python yue2_convert_lora.py --input my_artist.safetensors --output my_artist.hf.safetensors --to hf --branch nar
python yue2_convert_lora.py --input nar_lora_joint_v4.pt --output v4_native.safetensors --to native \
  --base_model yue2_3b_bf16.safetensors
python yue2_convert_lora.py --input my_artist.safetensors --output my_artist_full.comfy.safetensors --to comfy \
  --concat nar_lora_joint_v9_comfyui.safetensors
```

- `--to`: `native` (musubi), `comfy` (ComfyUI; NAR and I/O via the model loader, AR via the CLIP loader of a LoRA loader node), `hf` (yue2-lora-v1 for Starnodes / YuE2 Studio; one branch per file, so a joint LoRA needs `--branch ar` or `--branch nar`), `fl` (fl-yue2-lora-v1 for ComfyUI-FL-YuE2; writes `{stem}-ar` / `{stem}-nar` files).
- `--from` forces the input format (`auto` by default), `--branch` keeps `ar` or `nar` modules, `--concat` rank-concatenates companion adapters, `--base_model` is needed for Mothersuperior full I/O weights, `--dtype` sets the output dtype (fp32 default), and `--verify` reloads the output and compares every weight delta.
- Split q/k/v and gate/up LoRAs become exact block-diagonal fused pairs in the Comfy export (the scale is baked into `lora_up`).
- `--yue2_export_formats` in the trainer writes `{name}.comfy.safetensors`, `{name}.hf-{ar|nar}.safetensors` and `{name}.fl-{ar|nar}.safetensors` next to every saved LoRA, including `{output_name}-best`. They are removed together with their checkpoint by `--save_last_n_steps` / `--save_last_n_epochs`. `hf` is rejected with `--train_io full` or with included base weights that carry an I/O bias delta, and `fl` with `include_time_embedder`; a failed export is logged and training continues.

## Inference / 推論

```bash
python yue2_generate_music.py --dit /models/yue2_3b_bf16.safetensors \
  --style "dark synthwave, female vocals, 110 bpm" --lyrics_file lyrics.txt --cot off --seconds 60 \
  --lora_weight output/my_artist.safetensors --save_path out --save_artifacts
```

- Prompt: `--style` (alias `--prompt`), `--lyrics` / `--lyrics_file`, `--cot off|melody|full` (default `full`: the AR first plans an ABC score, then the song), `--abc` / `--abc_file` (use a given score), `--seed 831001`, `--seconds` (limits the length; 25 codes per second).
- Semantic sampling defaults follow the official protocol: `--temperature 1.0 --top_p 0.95 --top_k 100 --repetition_penalty 1.2 --penalty_window 50 --max_tokens 9000 --min_tokens 200`. `--cfg_scale` defaults to the protocol value (1.01 for cot off, else 1.0). ABC sampling uses `--abc_*`.
- NAR/VAE: `--ode_steps 32`, `--ode_state_dtype bf16`, `--nar_context codes|text_only`, `--vae_core_frames 1024 --vae_halo_frames 16`, `--vae_cpu`.
- Reconstruct instead of generating tokens: `--codes_file` (a `codes.safetensors` artifact, `.npy`, `.json`, `.txt`) or `--reconstruct_cache` (a YuE2 latent cache; also writes the ground truth).
- LoRA: `--lora_weight` (any format, several allowed), `--lora_multiplier`, `--lora_ar_multiplier`, `--lora_nar_multiplier`. LoRAs are merged at load. `--lora_attach` keeps them as modules, which is implied for the int8 file.
- Memory: `--convrot_int8`, `--fp8_scaled`, the int8 file, `--blocks_to_swap`, `--ar_blocks_to_swap`, `--block_swap_h2d_only`, `--sample_ar_resident`; `--attn_mode torch|sdpa|flash|flash3|xformers|sageattn`, `--split_attn`.
- Output: `--output_format flac|wav`, `--output_name`, `--save_artifacts` (ABC text, codes, latents, settings.json); `--from_file` (prompt file as above) or `--interactive`.

Measured on an H100 (30 s, cot off, SDPA): bf16 30 tokens/s, peak 8.8 GB; int8 file 12 tokens/s, 5.9 GB; bf16 with `--blocks_to_swap 14 --sample_ar_resident` 30 tokens/s, 6.2 GB. The generator does not use CUDA graphs; the CUDA-graph path of the official `yue2_infer` package decodes about 5x faster. With the same weights and seed, tokens and latents match the official `yue2_infer` exactly.

## Known limitations / 既知の制限

- **No demonstrated perceptual win.** No YuE2 LoRA has yet been shown to clearly beat the base model in listening tests. Validation losses and renders are there to pick checkpoints, not to prove quality.
- **Tokenizer dialect.** AR LoRAs learn the semantic head's codes (see above). Pair them with the matching companion NAR adapter.
- **Text-only NAR mismatch.** `--nar_context text_only` trains without the codes that inference provides.
- **Window vs whole song.** The NAR trains on windows (60 s by default) but renders the whole song in context-sized chunks. The AR trains only on segments that start at the song start.
- **Official values unknown.** The official codec-dropout rate and the NAR conditioning boundary are not published. The `[instrumental]` placeholder and the ABC flavour heuristics are community conventions; both are configurable and recorded in the metadata.
- Not included: SheetSage2 ABC transcription (scores must come from your own `.abc.txt` sidecars or JSONL fields), CUDA graphs and batched CFG in the AR sampler, audio-domain auxiliary losses, and LoHa/LoKr. flash_attn 2 and xformers are implemented but were not tested (only SDPA and flash3 were available on the test machine).

## Testing on real weights (developers) / 実重みでのテスト（開発者向け）

CPU tests: `pytest tests -k yue2`. The `qwen.tiktoken` backend tests (`tests/test_yue2_tokenizer.py` and the tokenizer checks of `tests/gpu/test_yue2_real_weights.py`) need `tiktoken` (in the `dev` dependency group) and skip without it. `tests/test_yue2_lora_v1_compat.py` reads `hf` and NAR `comfy` exports with an independent yue2-lora-v1 reader and merges them into the official model. GPU checks with released weights live in `tests/gpu/` and need `YUE2_WEIGHTS_DIR` (Hugging Face repo ids with `/` replaced by `__`, see `tests/gpu/conftest.py`) and `YUE2_DATA_DIR/jamendolyrics` (JamendoLyrics audio and lyrics):

- `pytest tests/gpu/test_yue2_real_weights.py -s`: loaders (HF == Comfy bf16 bit-exact, ConvRot from both sources identical, int8 dequantization error), tokenizer, AR and NAR parity with the official `yue2_infer` code, VAE parity, and the ComfyUI export key check. It skips cleanly without weights or CUDA.
- `python tests/gpu/yue2_train_matrix.py --out DIR --rows reduced|full|overfit|abc|NAME,...`: training runs across modes, precisions, memory settings and a synthetic ABC score (`abc`: `--cot full` with `--ar_ce_targets codec_abc` vs `codec`) with a JSON report (equivalence at step 1, quantized-vs-bf16 loss, determinism, resume state, overfit).
- `python tests/gpu/yue2_train_parity.py --cache_dir DIR/cache`: the trainer's flow, CE and KL losses against written-out loss formulas on the official model code (`tests/yue2_ref`: `nar_velocity` with its full AR/NAR attention mask, and the AR backbone).
- `python tests/gpu/yue2_comfy_export_check.py --checkpoint yue2_3b_bf16.safetensors [--lora X] [--comfy_root ComfyUI]`: maps a Comfy export onto the checkpoint; with `--comfy_root` it also loads the checkpoint and the LoRA through ComfyUI and compares every patched weight with the musubi delta. Checked with ComfyUI `95539f5` and a random joint LoRA (AR + NAR + I/O): 116 model patches and 112 CLIP patches (all expected keys), no unloaded keys, worst relative delta error 4.9e-6. Audio rendered in ComfyUI was not compared.
