# ACE-Step 1.5 LoRA Training

Train LoRA adapters for [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) music generation model.

## Quick Start

```bash
# 1. Download models (~10GB)
_ac15_download_models.bat

# 2. Cache text encoder outputs
_ac15_cache_text.bat

# 3. Cache audio latents
_ac15_cache_latents.bat

# 4. Train
_ac15_train.bat
```

## Requirements

- Python 3.10+
- CUDA GPU with 8GB+ VRAM (16GB+ recommended)
- ~10GB disk space for models
- Audio files (.wav, .mp3, .flac) with caption .txt files

Install dependencies:
```bash
pip install librosa vector_quantize_pytorch
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

Edit `_ac15_example_dataset.toml`:

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

## Commands

### Download Models

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='./checkpoints')"
```

### Cache Text Encoder Outputs

```bash
python acestep_cache_text_encoder_outputs.py ^
    --dataset_config _ac15_example_dataset.toml ^
    --text_encoder checkpoints/Qwen3-Embedding-0.6B ^
    --skip_existing
```

### Cache Audio Latents

```bash
python acestep_cache_latents.py ^
    --dataset_config _ac15_example_dataset.toml ^
    --vae checkpoints/vae ^
    --skip_existing
```

### Train LoRA

```bash
python acestep_train_network.py ^
    --dataset_config _ac15_example_dataset.toml ^
    --dit checkpoints/acestep-v15-turbo ^
    --output_dir output/acestep_lora ^
    --network_module networks.lora_acestep ^
    --network_dim 16 ^
    --network_alpha 16 ^
    --max_train_steps 2000 ^
    --learning_rate 1e-4 ^
    --mixed_precision bf16 ^
    --gradient_checkpointing ^
    --save_every_n_steps 500 ^
    --save_state
```

### Resume Training

```bash
python acestep_train_network.py ^
    --dataset_config _ac15_example_dataset.toml ^
    --dit checkpoints/acestep-v15-turbo ^
    --output_dir output/acestep_lora ^
    --network_module networks.lora_acestep ^
    --network_dim 16 ^
    --network_alpha 16 ^
    --max_train_steps 4000 ^
    --learning_rate 1e-4 ^
    --mixed_precision bf16 ^
    --gradient_checkpointing ^
    --save_state ^
    --resume output/acestep_lora/checkpoint-500
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--network_dim` | LoRA rank (higher = more capacity) | 16 |
| `--network_alpha` | LoRA alpha (typically same as dim) | 16 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--max_train_steps` | Total training steps | 2000 |
| `--save_every_n_steps` | Checkpoint interval | 500 |
| `--gradient_checkpointing` | Reduce VRAM usage | - |
| `--mixed_precision` | Must be `bf16` for ACE-Step | bf16 |

## Audio Sampling During Training

Add these flags to generate audio samples during training:

```bash
--vae checkpoints/vae ^
--text_encoder checkpoints/Qwen3-Embedding-0.6B ^
--sample_prompts _ac15_sample_prompts.json ^
--sample_every_n_steps 500
```

Sample prompts JSON format:

```json
[
    {
        "prompt": "Electronic dance music with heavy bass",
        "lyrics": "[Instrumental]",
        "audio_duration": 30.0
    }
]
```

## Tips

- **VRAM**: Use `--gradient_checkpointing` to reduce memory usage
- **LoRA dim**: Start with 16, increase to 32-64 for more complex styles
- **Steps**: 1000-2000 steps is usually enough for style fine-tuning
- **Captions**: Be descriptive - include genre, instruments, mood, tempo
- **Audio length**: 30-120 seconds works well, longer files use more memory

## Batch Files

| File | Purpose |
|------|---------|
| `_ac15_download_models.bat` | Download models from HuggingFace |
| `_ac15_cache_text.bat` | Cache text encoder outputs |
| `_ac15_cache_latents.bat` | Cache audio latents |
| `_ac15_train.bat` | Train LoRA |
| `_ac15_train_resume.bat` | Resume from checkpoint |

## Output

Trained LoRA weights are saved to:
- `output/acestep_lora/` - LoRA safetensors files
- `output/acestep_lora/checkpoint-*/` - Full training state (for resuming)
