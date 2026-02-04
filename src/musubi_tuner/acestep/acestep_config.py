"""
ACE-Step 1.5 configuration constants.

Audio processing and training configuration for ACE-Step model integration.
"""

# Audio processing constants
ACESTEP_SAMPLE_RATE = 48000  # 48kHz
ACESTEP_CHANNELS = 2  # Stereo
ACESTEP_LATENT_HZ = 25  # Latent time resolution (48000 / 1920 = 25)
ACESTEP_LATENT_CHANNELS = 64  # VAE latent dimension
ACESTEP_VAE_TEMPORAL_FACTOR = 1920  # Audio samples per latent frame

# Duration limits
ACESTEP_MIN_DURATION_SECONDS = 5.0
ACESTEP_MAX_DURATION_SECONDS = 240.0  # 4 minutes

# Turbo model discrete timesteps (shift=3.0, 8 steps)
# These match the inference schedule exactly
TURBO_SHIFT3_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
]

# Default LoRA configuration
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1

# Text encoder configuration
TEXT_ENCODER_MAX_LENGTH = 256

# SFT format template for text encoding (must end with <|endoftext|> like official)
SFT_GEN_PROMPT = """# Instruction
{0}

# Caption
{1}

# Metas
{2}<|endoftext|>
"""
