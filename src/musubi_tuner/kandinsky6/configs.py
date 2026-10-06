from __future__ import annotations

LITE_CONFIG = {
    "audio_freqs_scaling": 0.144, "axes_dims": (16, 24, 24), "axes_dims_a": (16, 24, 24),
    "ca_rope": True, "cross_gates": True, "ff_dim": 7168, "ff_dim_a": 3584,
    "fix_modulation": True, "in_audio_dim": 40, "in_text_dim": 3584, "in_text_dim2": 768,
    "in_visual_dim": 16, "is_multimodal": True, "model_dim": 1792, "model_dim_a": 896,
    "num_text_blocks": 2, "num_visual_blocks": 32, "out_audio_dim": 40, "out_visual_dim": 16,
    "patch_size": (1, 2, 2), "text_token_padding": True, "time_dim": 512, "time_dim_a": 512,
    "visual_cond": True, "visual_token_type_num_embeddings": 2,
}

PRO_CONFIG = {
    **LITE_CONFIG, "axes_dims": (32, 48, 48), "axes_dims_a": (32, 48, 48), "ff_dim": 16384,
    "ff_dim_a": 7168, "model_dim": 4096, "model_dim_a": 2048, "num_text_blocks": 4,
    "num_visual_blocks": 60, "time_dim": 1024, "time_dim_a": 1024,
}

MODEL_CONFIGS = {"lite": LITE_CONFIG, "pro": PRO_CONFIG}
