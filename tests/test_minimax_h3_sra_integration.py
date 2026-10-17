from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.block_sparse_attention import BlockSparseConfig
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.sparse_attention import H3ReferenceSparseConfig
from musubi_tuner.minimax_h3.sparse_attention import H3UniversalSparseRouter
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


def _tiny_transformer() -> MiniMaxH3Transformer:
    return MiniMaxH3Transformer(
        MiniMaxH3TransformerConfig(
            num_attention_heads=1,
            attention_head_dim=128,
            hidden_size=128,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=256,
            in_channels=1,
            audio_in_channels=2,
            patch_size=(1, 1, 1),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=8,
        )
    )


def test_sra_cli_defaults_off_and_metadata_records_configuration() -> None:
    args = create_parser().parse_args([])
    assert args.h3_sparse_router is None
    assert args.h3_sparse_exact_blocks == 64
    assert args.h3_sparse_temporal_radius == 2
    assert args.h3_sparse_data_path == "packed"

    metadata = MiniMaxH3NetworkTrainer().extra_metadata(args)
    assert metadata["ss_h3_sparse_reference_attention"] == "False"
    assert metadata["ss_h3_sparse_exact_blocks"] == "64"
    assert metadata["ss_h3_sparse_temporal_radius"] == "2"
    assert metadata["ss_h3_sparse_data_path"] == "packed"


def test_sra_rejects_other_attention_backends() -> None:
    transformer = _tiny_transformer()
    config = H3ReferenceSparseConfig(layer_ranges=((0, 2),))
    transformer.set_block_sparse_attention(BlockSparseConfig(kv_fraction=0.5))
    with pytest.raises(ValueError, match="mutually exclusive"):
        transformer.enable_reference_sparse_attention(config)

    transformer = _tiny_transformer()
    transformer.enable_reference_sparse_attention(config)
    with pytest.raises(ValueError, match="cannot be combined"):
        transformer.set_int8_attention_mode("train")


def test_sra_router_checkpoint_rejects_layer_geometry_before_model_mutation(tmp_path) -> None:
    checkpoint = tmp_path / "router.pt"
    torch.save(
        {
            "query_layer_projection": torch.randn(3, 128, 4),
            "key_layer_projection": torch.randn(3, 128, 4),
        },
        checkpoint,
    )
    args = create_parser().parse_args(["--h3_sparse_router", str(checkpoint)])
    trainer = MiniMaxH3NetworkTrainer()
    trainer._crepa_config = None
    transformer = _tiny_transformer()

    with pytest.raises(ValueError, match="3 layers; model has 2"):
        trainer.on_transformer_loaded(args, SimpleNamespace(), transformer)
    assert transformer.reference_sparse_config is None


def test_sra_router_checkpoint_loads_frozen_with_requested_layout(tmp_path) -> None:
    checkpoint = tmp_path / "router.pt"
    router = H3UniversalSparseRouter(
        num_layers=2,
        num_heads=1,
        head_dim=128,
        index_dim=4,
        block_size=128,
        layer_specific_projections=True,
    )
    torch.save(router.state_dict(), checkpoint)
    args = create_parser().parse_args(
        ["--h3_sparse_router", str(checkpoint), "--h3_sparse_exact_blocks", "3", "--h3_sparse_data_path", "mapped_int64"]
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer._crepa_config = None
    trainer.num_timestep_buckets = None
    trainer.timestep_range_pool = []
    accelerator = SimpleNamespace(
        is_main_process=True,
        register_save_state_pre_hook=lambda hook: None,
        register_load_state_pre_hook=lambda hook: None,
    )
    transformer = _tiny_transformer()

    trainer.on_transformer_loaded(args, accelerator, transformer)

    assert transformer.reference_sparse_config.topk_blocks == 3
    assert transformer.reference_sparse_config.data_path == "mapped_int64"
    assert transformer.reference_sparse_router.training is False
    assert all(not parameter.requires_grad for parameter in transformer.reference_sparse_router.parameters())
