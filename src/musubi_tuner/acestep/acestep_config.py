"""
ACE-Step configuration constants.

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

# Default inference settings per model type
BASE_MODEL_DEFAULTS = {
    "shift": 1.0,
    "guidance_scale": 7.0,
    "inference_steps": 50,
}

TURBO_MODEL_DEFAULTS = {
    "shift": 3.0,
    "guidance_scale": 0.0,
    "inference_steps": 8,
}


def compute_shifted_timesteps(num_steps: int, shift: float) -> list:
    """Compute discrete timestep schedule with shift applied.

    Base timesteps: t_i = 1 - i/N for i in 0..N-1
    Shifted: t_shifted = shift * t / (1 + (shift - 1) * t)
    """
    timesteps = []
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        if shift != 1.0:
            t = shift * t / (1.0 + (shift - 1.0) * t)
        timesteps.append(t)
    return timesteps

# FP8 optimization target/exclude patterns (for load_safetensors_with_lora_and_fp8)
ACESTEP_FP8_TARGET_KEYS = ["decoder"]
ACESTEP_FP8_EXCLUDE_KEYS = [
    "encoder",
    "norm",
    "_emb",
    "null_condition",
    "embedding",
    # Decoder conv projections are non-Linear; keep them in bf16/fp16.
    "proj_in.1",
    "proj_out.1",
]

# Default LoRA configuration
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1

# Text encoder configuration
TEXT_ENCODER_MAX_LENGTH = 256
ACESTEP_TEXT_CACHE_SCHEMA_VERSION = "2"

# Default instruction - must match original ACE-Step trainer
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# SFT format template for text encoding (must end with <|endoftext|> like official)
SFT_GEN_PROMPT = """# Instruction
{0}

# Caption
{1}

# Metas
{2}<|endoftext|>
"""
