from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy

from musubi_tuner.minimax_h3 import selective_checkpoint as sc
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser

# The op SDPA dispatches for float32 CPU inputs, which is what the tiny model hits.
_CPU_ATTENTION = "aten::_scaled_dot_product_flash_attention_for_cpu"
_LORA_RANK = 4


def _tiny_config(num_layers: int = 2) -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=num_layers,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


def _tiny_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return {
        "video_hidden_states": torch.randn(1, 2, 16, generator=generator),
        "audio_hidden_states": torch.randn(1, 4, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, 3, 12, generator=generator),
        "timestep": torch.tensor([0.25, 0.75]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0]),
        "position_ids": torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10,
        "video_indices": torch.tensor([7, 8]),
        "audio_indices": torch.tensor([3, 4, 5, 6]),
        "text_indices": torch.tensor([0, 1, 2]),
    }


class _OpCounter(TorchDispatchMode):
    """Counts the ops that actually execute; cached replays never reach an outer mode."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[sc.op_name(func)] += 1
        return func(*args, **(kwargs or {}))


def _wrap_qkv_with_lora(model: MiniMaxH3Transformer) -> None:
    """Replace each ``qkv_proj.forward`` the way ``LoRAModule.apply_to`` does, after the keep mode is set."""
    generator = torch.Generator().manual_seed(5)
    for block in model.blocks:
        projection = block.attn.qkv_proj
        down = torch.nn.Linear(projection.in_features, _LORA_RANK, bias=False)
        up = torch.nn.Linear(_LORA_RANK, projection.out_features, bias=False)
        down.weight.data = torch.randn(down.weight.shape, generator=generator) * 0.1
        up.weight.data = torch.randn(up.weight.shape, generator=generator) * 0.1
        block.attn.lora_down = down
        block.attn.lora_up = up
        org_forward = projection.forward
        projection.forward = lambda x, org_forward=org_forward, down=down, up=up: org_forward(x) + up(down(x))


def _run(
    keep: str, *, checkpointing: bool = True, lora: bool = False
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Counter[str], sc.SavedActivations]:
    torch.manual_seed(11)
    model = MiniMaxH3Transformer(_tiny_config())
    if checkpointing:
        model.enable_gradient_checkpointing()
        model.set_checkpoint_keep(keep)
    if lora:
        _wrap_qkv_with_lora(model)
    counter = _OpCounter()
    with counter:
        output = model(**_tiny_inputs())
        loss = output.video.square().mean() + output.audio.square().mean()
        loss.backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    return loss.detach(), grads, counter.counts, model._checkpoint_saved


@pytest.mark.parametrize("lora", [False, True], ids=["plain", "lora"])
@pytest.mark.parametrize("keep", ["attention", "qkv"])
def test_keeping_activations_is_bit_identical_to_plain_checkpointing(keep, lora):
    num_blocks = _tiny_config().num_layers
    reference_loss, reference_grads, plain, _ = _run("none", lora=lora)
    loss, grads, kept, saved = _run(keep, lora=lora)

    assert reference_grads
    if lora:
        assert any("lora_down" in name for name in reference_grads)
    assert torch.equal(loss, reference_loss)
    assert grads.keys() == reference_grads.keys()
    for name, reference in reference_grads.items():
        assert torch.equal(grads[name], reference), name
    # Plain checkpointing runs every main-block attention forward twice; a kept
    # output is replayed from the cache instead. The refiner block is untouched.
    assert plain[_CPU_ATTENTION] - kept[_CPU_ATTENTION] == num_blocks
    assert saved.attention == num_blocks
    if keep == "qkv":
        # Exactly the base projection matmul per block is kept; under LoRA the
        # down/up matmuls of the adapter are still recomputed.
        assert saved.projection == num_blocks
        _, _, attention_only, _ = _run("attention", lora=lora)
        assert attention_only["aten::mm"] - kept["aten::mm"] == num_blocks
    else:
        assert saved.projection == 0
        assert plain["aten::mm"] == kept["aten::mm"]


def test_plain_checkpointing_matches_no_checkpointing():
    eager_loss, eager_grads, _, _ = _run("none", checkpointing=False)
    loss, grads, _, _ = _run("none")
    assert torch.equal(loss, eager_loss)
    for name, reference in eager_grads.items():
        assert torch.equal(grads[name], reference), name


def test_a_pass_whose_attention_was_not_a_saved_op_is_refused(monkeypatch):
    # Stand-in for SDPA resolving to the math backend: no fused op reaches the policy.
    monkeypatch.setattr(sc, "ATTENTION_OUTPUT_OPS", frozenset())
    model = MiniMaxH3Transformer(_tiny_config())
    model.enable_gradient_checkpointing()
    model.set_checkpoint_keep("attention")
    with pytest.raises(RuntimeError, match="saved 0 fused attention outputs for 2 checkpointed block calls"):
        model(**_tiny_inputs())
    # No-grad passes checkpoint nothing and are not judged.
    with torch.no_grad():
        model(**_tiny_inputs())


def test_a_pass_whose_projection_bypassed_the_region_is_refused():
    model = MiniMaxH3Transformer(_tiny_config())
    model.enable_gradient_checkpointing()
    model.set_checkpoint_keep("qkv")
    # A fused adapter kernel replaces the projection call without ever
    # entering the base forward the region wraps.
    for block in model.blocks:
        projection = block.attn.qkv_proj
        projection.forward = lambda x, weight=projection.weight: torch.nn.functional.linear(x, weight.detach())
    with pytest.raises(RuntimeError, match="saved 0 QKV projection outputs"):
        model(**_tiny_inputs())


def test_policy_saves_only_the_attention_kernels():
    ctx = SimpleNamespace(is_recompute=False)
    saved = sc.SavedActivations()
    for name in sorted(sc.ATTENTION_OUTPUT_OPS):
        namespace, op = name.split("::")
        try:
            func = getattr(getattr(torch.ops, namespace), op).default
        except (AttributeError, RuntimeError):
            continue  # flash-attn ops exist only where the package is installed
        assert sc.checkpoint_policy("attention", saved, ctx, func) is CheckpointPolicy.MUST_SAVE, name
        assert sc.checkpoint_policy("qkv", saved, ctx, func) is CheckpointPolicy.MUST_SAVE, name
    flash = torch.ops.aten._scaled_dot_product_flash_attention.default
    assert sc.checkpoint_policy("attention", saved, ctx, flash) is CheckpointPolicy.MUST_SAVE
    assert saved.attention > 0
    # Recompute replays are not counted a second time.
    before = saved.attention
    assert sc.checkpoint_policy("attention", saved, SimpleNamespace(is_recompute=True), flash) is CheckpointPolicy.MUST_SAVE
    assert saved.attention == before
    a = torch.zeros(9, 32)
    b = torch.zeros(32, 96)
    for func in (torch.ops.aten.mm.default, torch.ops.aten.softmax.int, torch.ops.aten.rsqrt.default, torch.ops.aten.bmm.default):
        assert sc.checkpoint_policy("attention", saved, ctx, func, a, b) is CheckpointPolicy.PREFER_RECOMPUTE
        assert sc.checkpoint_policy("qkv", saved, ctx, func, a, b) is CheckpointPolicy.PREFER_RECOMPUTE
    assert saved.projection == 0


def test_qkv_policy_saves_the_base_projection_matmul_only():
    ctx = SimpleNamespace(is_recompute=False)
    hidden = torch.zeros(9, 32)
    weight_t = torch.zeros(32, 96)
    mm = torch.ops.aten.mm.default
    with sc.projection_region(32, 96):
        assert sc.checkpoint_policy("qkv", None, ctx, mm, hidden, weight_t) is CheckpointPolicy.MUST_SAVE
        assert (
            sc.checkpoint_policy("qkv", None, ctx, torch.ops.aten.addmm.default, torch.zeros(96), hidden, weight_t)
            is CheckpointPolicy.MUST_SAVE
        )
        # A LoRA down-projection reduces over the same width but is not the
        # projection; an up-projection reduces over the rank. Neither is kept.
        assert sc.checkpoint_policy("qkv", None, ctx, mm, hidden, torch.zeros(32, 4)) is CheckpointPolicy.PREFER_RECOMPUTE
        assert (
            sc.checkpoint_policy("qkv", None, ctx, mm, torch.zeros(9, 4), torch.zeros(4, 96)) is CheckpointPolicy.PREFER_RECOMPUTE
        )
        # A ConvRot group rotation is a matmul over the group width.
        assert (
            sc.checkpoint_policy("qkv", None, ctx, torch.ops.aten.bmm.default, torch.zeros(4, 9, 8), torch.zeros(4, 8, 8))
            is CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert (
            sc.checkpoint_policy("qkv", None, ctx, torch.ops.aten.add.Tensor, hidden, hidden) is CheckpointPolicy.PREFER_RECOMPUTE
        )
        # The attention level ignores the region entirely.
        assert sc.checkpoint_policy("attention", None, ctx, mm, hidden, weight_t) is CheckpointPolicy.PREFER_RECOMPUTE
    # Outside the region the same matmul is an out-projection or a feed-forward.
    assert sc.checkpoint_policy("qkv", None, ctx, mm, hidden, weight_t) is CheckpointPolicy.PREFER_RECOMPUTE
    assert sc._projection_shape.get() is None


def test_projection_region_is_installed_once_and_gated():
    model = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    attention = model.blocks[0].attn
    original_forward = attention.qkv_proj.forward
    model.set_checkpoint_keep("qkv")
    wrapped = attention.qkv_proj.forward
    assert wrapped is not original_forward
    model.set_checkpoint_keep("qkv")
    assert attention.qkv_proj.forward is wrapped
    seen: list[tuple[int, int] | None] = []
    hidden = torch.randn(1, 3, 32)
    original = torch.ops.aten.mm.default

    class _Probe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is original:
                seen.append(sc._projection_shape.get())
            return func(*args, **(kwargs or {}))

    with _Probe():
        attention.qkv_proj(hidden)
    assert seen == [(32, 96)]
    model.set_checkpoint_keep("none")
    assert attention.qkv_proj.forward is wrapped
    seen.clear()
    with _Probe():
        attention.qkv_proj(hidden)
    assert seen == [None]


def test_attention_kernel_visibility_and_mode_validation():
    assert sc.attention_kernel_is_visible("torch")
    if not sc._op_is_registered("flash_attn", "_flash_attn_forward"):
        assert not sc.attention_kernel_is_visible("flash")
        model = MiniMaxH3Transformer(_tiny_config(num_layers=1), attention_mode="flash")
        with pytest.raises(ValueError, match="cannot see the 'flash' attention kernel"):
            model.set_checkpoint_keep("attention")
    with pytest.raises(ValueError, match="must be one of"):
        sc.validate_checkpoint_keep("all")
    with pytest.raises(ValueError, match="needs no context_fn"):
        sc.checkpoint_context_fn("none")


def test_model_rejects_keep_with_cpu_offload_in_either_order():
    model = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    model.enable_gradient_checkpointing(activation_cpu_offloading=True)
    with pytest.raises(ValueError, match="CPU offload"):
        model.set_checkpoint_keep("attention")
    model = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    model.set_checkpoint_keep("attention")
    with pytest.raises(ValueError, match="CPU offload"):
        model.enable_gradient_checkpointing(activation_cpu_offloading=True)
    assert all(block.attn.checkpoint_keep_qkv is False for block in model.blocks)
    model.set_checkpoint_keep("qkv")
    assert all(block.attn.checkpoint_keep_qkv for block in model.blocks)
    model.set_checkpoint_keep("none")
    assert all(block.attn.checkpoint_keep_qkv is False for block in model.blocks)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--h3_checkpoint_keep", "attention"], "requires --gradient_checkpointing"),
        (
            ["--gradient_checkpointing", "--gradient_checkpointing_cpu_offload", "--h3_checkpoint_keep", "attention"],
            "gradient_checkpointing_cpu_offload",
        ),
        (["--gradient_checkpointing", "--compile", "--h3_checkpoint_keep", "qkv"], "--compile"),
        (["--gradient_checkpointing", "--h3_int8_attention", "train", "--h3_checkpoint_keep", "attention"], "int8_attention train"),
        (["--gradient_checkpointing", "--h3_block_sparse_kv_fraction", "0.5", "--h3_checkpoint_keep", "attention"], "block-sparse"),
        (
            [
                "--gradient_checkpointing",
                "--blocks_to_swap",
                "4",
                "--block_swap_h2d_only",
                "--block_swap_granularity",
                "layer",
                "--h3_checkpoint_keep",
                "attention",
            ],
            "block_swap_granularity layer",
        ),
        (
            [
                "--gradient_checkpointing",
                "--h3_convrot_int8",
                "--h3_convrot_int8_fwd",
                "int8",
                "--h3_convrot_int8_bwd",
                "int8",
                "--h3_convrot_int8_lora_fused",
                "--h3_checkpoint_keep",
                "qkv",
            ],
            "h3_convrot_int8_lora_fused",
        ),
    ],
)
def test_trainer_rejects_incompatible_keep_combinations(argv, message):
    args = create_parser().parse_args(["--sdpa", *argv])
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_trainer_accepts_keep_with_block_swap_and_wires_it_to_the_model():
    args = create_parser().parse_args(
        ["--sdpa", "--gradient_checkpointing", "--blocks_to_swap", "4", "--block_swap_h2d_only", "--h3_checkpoint_keep", "qkv"]
    )
    assert create_parser().parse_args(["--sdpa"]).h3_checkpoint_keep == "none"
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    model = MiniMaxH3Transformer(_tiny_config(num_layers=1))
    accelerator = SimpleNamespace(
        is_main_process=True,
        device=torch.device("cpu"),
        register_save_state_pre_hook=lambda hook: None,
        register_load_state_pre_hook=lambda hook: None,
    )
    trainer.on_transformer_loaded(args, accelerator, model)
    assert model.checkpoint_keep == "qkv"
    assert model.blocks[0].attn.checkpoint_keep_qkv
    assert getattr(model.blocks[0].attn.qkv_proj, "_h3_projection_region", False)
