"""Constants for the Kandinsky 5 Super Resolution dataset."""

# Compression factors are a property of the encoding VAE, keyed by the
# ``vae.name`` config value accepted by the VAE builders.
VAE_FACTORS: dict[str, tuple[int, int]] = {
    # name: (spatial downsample, temporal downsample)
    "video-kvae": (16, 4),
}

VAE_SPATIAL_FACTOR: int = 16
VAE_TEMPORAL_FACTOR: int = 4


def set_vae_factors(vae_name: str) -> None:
    """Set the process-wide VAE compression factors from the VAE name.

    ``load_sr_components`` calls this once at startup (before any
    latent<->pixel size math) with the training config's ``vae.name``.

    Consumers must read the values late-bound as
    ``constants.VAE_SPATIAL_FACTOR`` / ``constants.VAE_TEMPORAL_FACTOR``
    (module attributes) — a ``from``-import freezes the defaults at
    import time.

    Args:
        vae_name: VAE name from the config (``vae.name``).

    Raises:
        ValueError: If the name is not in :data:`VAE_FACTORS`.
    """
    global VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR
    if vae_name not in VAE_FACTORS:
        msg = f"Unknown VAE name {vae_name!r}; known: {sorted(VAE_FACTORS)}"
        raise ValueError(msg)
    VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR = VAE_FACTORS[vae_name]


RESOLUTIONS: dict[int, list[tuple[int, int]]] = {
    512: [(512, 512), (512, 768), (768, 512)],
}


# --- Model / pipeline contract (NOT configurable) ---
# The SR model was trained on 5 s clips at 24 fps -> 121 pixel frames
# (= 1 + 8*15); feeding more frames or another rate is out of contract.
TARGET_FPS: int = 24
MAX_NUM_FRAMES: int = 121
# Treat ``|src_fps - TARGET_FPS| < tolerance`` as already-at-target (no
# resample): covers 23.98 (NTSC) and 25.00 sources.
RESAMPLE_FPS_TOLERANCE: float = 1.5
# Expected rank of a raw ``[T, C, H, W]`` LR latent tensor.
LATENT_NDIM: int = 4
# Defensive dynamo recompile ceiling for the nabla flex_attention path.
SPARSE_RECOMPILE_LIMIT: int = 256
# Frames per interpolate call when downscaling the output (a 2K clip is
# ~2.4 GB as uint8, so the resize walks the clip in chunks).
RESIZE_FRAME_CHUNK: int = 8
# Delivery tiers for --target-resolution: (H, W) candidates per tier —
# landscape 16:9, square, portrait 9:16; the closest-aspect entry wins.
TARGET_RESOLUTIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "hd": ((720, 1280), (720, 720), (1280, 720)),
    "fullhd": ((1080, 1920), (1080, 1080), (1920, 1080)),
    "2k": ((1440, 2560), (1440, 1440), (2560, 1440)),
}
