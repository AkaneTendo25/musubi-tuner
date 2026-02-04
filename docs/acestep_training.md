# ACE-Step 1.5 LoRA Training

Train LoRA adapters for [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) music generation model.

## Requirements

- Python 3.10+
- CUDA GPU with 12GB+ VRAM (24GB+ recommended for larger batches)
- ~10GB disk space for models
- Audio files (.wav, .mp3, .flac) with metadata .json files

### Installation

First, install musubi-tuner as usual (see main README), then add ACE-Step specific dependencies:

```bash
# Additional dependencies for ACE-Step branch
pip install vector_quantize_pytorch
```

For mp3 file support (optional):
```bash
pip install librosa
```

For flash attention (recommended for faster training):
```bash
pip install flash-attn --no-build-isolation
```

## Captioning Guide

Based on official ACE-Step guidelines. Caption dimensions:

| Dimension | Examples |
|-----------|----------|
| Style/Genre | pop, rock, jazz, electronic, hip-hop, lo-fi, synthwave |
| Emotion | melancholic, uplifting, energetic, dreamy, dark, euphoric |
| Instruments | acoustic guitar, piano, synth pads, 808 drums, strings |
| Timbre | warm, bright, crisp, airy, punchy, lush, raw, polished |
| Era | 80s synth-pop, 90s grunge, 2010s EDM, vintage soul |
| Production | lo-fi, high-fidelity, live recording, studio-polished |
| Vocals | female vocal, male vocal, breathy, powerful, raspy |
| Tempo | slow tempo, mid-tempo, fast-paced, groovy, driving |

**Principles:**
- Specific beats vague ("sad piano ballad with female breathy vocal" > "a sad song")
- Combine dimensions (style + emotion + instruments + timbre)
- Texture words affect mixing (warm, crisp, airy, punchy)
- Avoid conflicting styles in same caption
- Don't include BPM/key in caption (use metadata parameters instead)

**Lyrics structure tags:**
- `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`
- `[Instrumental]`, `[Guitar Solo]`, `[Piano Interlude]`
- `[Build]`, `[Drop]`, `[Breakdown]`, `[Fade Out]`

See [ACE-Step Tutorial](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) for complete guide.

## Dataset Setup

Create a folder with your audio files and matching JSON metadata files:

```
training_data/
├── song1.wav
├── song1.json     # Metadata (caption, lyrics, bpm, key, etc.)
├── song2.mp3
├── song2.json
└── ...
```

### JSON Metadata Format

Each audio file requires a `.json` file with the same name:

```json
{
    "caption": "Electronic dance music with heavy bass and synth leads, energetic and driving",
    "lyrics": "[Intro]\n[Synth arpeggio]\n\n[Verse]\nDancing in the night...\n\n[Chorus]\nFeel the beat drop",
    "bpm": 128,
    "keyscale": "F minor",
    "timesignature": "4",
    "duration": 180
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `caption` | Yes | Detailed music description (style, instruments, mood, etc.) |
| `lyrics` | No | Song lyrics with structure tags, or `"[Instrumental]"` |
| `bpm` | No | Beats per minute |
| `keyscale` | No | Musical key (e.g., "C major", "F# minor") |
| `timesignature` | No | Time signature (e.g., "4", "3", "6") |
| `duration` | No | Duration in seconds |

The metadata is formatted into the SFT prompt that ACE-Step was trained on, providing stronger conditioning for better learning.

### Dataset Config

Create dataset config (e.g., `acestep_dataset.toml`):

```toml
[general]
caption_extension = ".json"

[[datasets]]
audio_directory = "path/to/your/training_data"
cache_directory = "path/to/cache"
batch_size = 1
num_repeats = 10
max_duration = 120.0
min_duration = 5.0
```

## Quick Start

### 1. Download Models (~10GB)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='./checkpoints')"
```

The DiT checkpoint must include `silence_latent.pt` (for example:
`checkpoints/acestep-v15-turbo/silence_latent.pt`). It is required for the
original ACE-Step text2music training flow.

### 2. Cache Text Encoder Outputs

```bash
python acestep_cache_text_encoder_outputs.py \
    --dataset_config acestep_dataset.toml \
    --text_encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo \
    --batch_size 8
```

`--dit` is recommended: it caches ACE-Step condition-encoder outputs (closer to the original trainer flow).

### 3. Cache Audio Latents

```bash
python acestep_cache_latents.py \
    --dataset_config acestep_dataset.toml \
    --vae checkpoints/vae \
    --max_duration 120 \
    --batch_size 1
```

For long audio files with limited VRAM, use chunked encoding:

```bash
python acestep_cache_latents.py \
    --dataset_config acestep_dataset.toml \
    --vae checkpoints/vae \
    --max_duration 240 \
    --vae_chunk_seconds 30 \
    --vae_chunk_overlap 2.0
```

Chunking uses overlapping segments with cosine crossfade blending for seamless results.

### 4. Train LoRA

With flash attention (recommended):
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 acestep_train_network.py \
    --dataset_config acestep_dataset.toml \
    --dit checkpoints/acestep-v15-turbo \
    --output_dir output/acestep_lora \
    --network_module networks.lora_acestep \
    --network_dim 64 \
    --network_alpha 128 \
    --network_dropout 0.1 \
    --optimizer_type AdamW8Bit \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --log_config \
    --log_with tensorboard \
    --logging_dir logs \
    --max_data_loader_n_workers 1 \
    --max_train_steps 2000 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --save_every_n_steps 500 \
    --save_state
```
## Audio Sampling During Training

Add these flags to generate audio samples during training:

```bash
    --vae checkpoints/vae \
    --text_encoder checkpoints/Qwen3-Embedding-0.6B \
    --sample_prompts sample_prompts.json \
    --sample_every_n_steps 500 \
    --sample_at_first
```

Use `--sample_at_first` to generate a sample before training starts (useful for verifying setup).

Sample prompts JSON format:

```json
[
    {
        "prompt": "Electronic dance music with heavy bass and synth leads",
        "lyrics": "[Instrumental]",
        "bpm": 128,
        "keyscale": "F minor",
        "timesignature": "4",
        "duration": 30.0,
        "audio_duration": 30.0,
        "seed": 42
    },
    {
        "prompt": "Acoustic folk song with warm guitar",
        "lyrics": "[Verse]\nSinging in the morning light\n[Chorus]\nOh the world is bright",
        "bpm": 95,
        "keyscale": "G major",
        "audio_duration": 30.0
    }
]
```

Supported fields:
- `prompt` (required): Music description
- `lyrics`: Song lyrics or `"[Instrumental]"` for instrumental tracks
- `bpm`: Beats per minute (optional, shows as "N/A" if omitted)
- `keyscale`: Musical key (optional, e.g., "C major", "F# minor")
- `timesignature`: Time signature (optional, e.g., "4", "3")
- `duration`: Duration for metadata formatting (optional)
- `audio_duration`: Actual generation duration in seconds (default: 30.0)
- `seed`: Random seed for reproducible generation (optional, random if omitted)

## LoRA Configuration

Official ACE-Step LoRA settings:

- `--network_dim 64` - LoRA rank
- `--network_alpha 128` - scaling factor (alpha/dim = 2.0)
- `--network_dropout 0.1` - dropout for regularization

The scale `alpha/dim` controls LoRA influence. Higher = stronger effect.

## Tips

- Use `--flash_attn` for faster training
- Use `--gradient_checkpointing` to reduce VRAM (~40%)
- Use `--lr_scheduler cosine --lr_warmup_steps 100`
- Use `--vae_chunk_seconds 30` for long files with limited VRAM
- 1000-2000 steps is usually enough
- Audio length 30-120 seconds works well
