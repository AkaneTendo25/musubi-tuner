# ACE-Step 1.5 LoRA Training

Train LoRA adapters for [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) music generation model.

## Requirements

- Python 3.10+
- CUDA GPU with 12GB+ VRAM (24GB+ recommended for larger batches)
- ~10GB disk space for models
- Audio files (.wav, .mp3, .flac) with caption .txt files

Install dependencies:
```bash
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

## Dataset Setup

Create a folder with your audio files and matching caption files:

```
training_data/
├── song1.wav
├── song1.txt      # Caption describing the music
├── song2.mp3
├── song2.txt
└── ...
```

Create dataset config (e.g., `acestep_dataset.toml`):

```toml
[general]
caption_extension = ".txt"

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

### 4. Train LoRA

With flash attention (recommended):
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 acestep_train_network.py \
    --dataset_config acestep_dataset.toml \
    --dit checkpoints/acestep-v15-turbo \
    --output_dir output/acestep_lora \
    --network_module networks.lora_acestep \
    --network_dim 16 \
    --network_alpha 16 \
    --optimizer_type AdamW8Bit \
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
        "audio_duration": 30.0
    },
    {
        "prompt": "Acoustic folk song with warm guitar",
        "lyrics": "[Verse]\nSinging in the morning light\n[Chorus]\nOh the world is bright",
        "audio_duration": 30.0
    }
]
```

Supported fields:
- `prompt` (required): Music description
- `lyrics`: Song lyrics or `"[Instrumental]"` for instrumental tracks
- `audio_duration`: Duration in seconds (default: 30.0)

## Tips

- **Flash Attention**: Use `--flash_attn` for ~30% faster training (requires `pip install flash-attn`)
- **VRAM**: Use `--gradient_checkpointing` to reduce memory usage (~40% reduction)
- **LoRA dim**: Start with 16, increase to 32-64 for more complex styles
- **Steps**: 1000-2000 steps is usually enough for style fine-tuning
- **Captions**: Be descriptive - include genre, instruments, mood, tempo
- **Audio length**: 30-120 seconds works well, longer files use more memory
- **Learning rate**: 1e-4 to 5e-5 work well for most cases
