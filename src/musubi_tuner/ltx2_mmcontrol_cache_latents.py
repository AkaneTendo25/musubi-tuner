"""MMControl latent caching entry point for LTX-2.

The cache implementation is shared with the existing VACE control cache path.
MMControl dataset config aliases are normalized by ``dataset.config_utils``.
"""

from __future__ import annotations

from musubi_tuner.ltx2_cache_latents import main


if __name__ == "__main__":
    main()
