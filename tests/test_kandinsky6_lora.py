from musubi_tuner.kandinsky6 import DiffusionTransformer3D, LITE_CONFIG
from musubi_tuner.networks.lora_kandinsky6 import create_arch_network, create_arch_network_from_weights


def test_lora_targets_both_fused_modalities():
    config = dict(LITE_CONFIG)
    config.update(
        model_dim=64,
        model_dim_a=64,
        ff_dim=128,
        ff_dim_a=64,
        time_dim=32,
        time_dim_a=32,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(16, 24, 24),
        axes_dims_a=(16, 24, 24),
        in_text_dim=48,
        in_text_dim2=16,
    )
    model = DiffusionTransformer3D(**config)
    network = create_arch_network(1.0, 4, 4.0, None, [], model)
    names = [module.lora_name for module in network.unet_loras]
    assert any("video_dec_block" in name for name in names)
    assert any("audio_dec_block" in name for name in names)
    assert any("va_cross_attention" in name for name in names)
    assert any("av_cross_attention" in name for name in names)

    state = network.state_dict()
    restored = create_arch_network_from_weights(1.0, state, unet=model)
    restored.load_state_dict(state, strict=True)
    assert set(restored.state_dict()) == set(state)
