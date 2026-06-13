import torch
from musubi_tuner.ltx_2.model.model_protocol import ModelConfigurator
from musubi_tuner.ltx_2.model.transformer.attention import Attention
from musubi_tuner.ltx_2.model.transformer.feed_forward import FeedForward
from musubi_tuner.ltx_2.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from musubi_tuner.ltx_2.utils import rms_norm


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """
    Embeddings1DConnector applies a 1D transformer-based processing to sequential embeddings (e.g., for video, audio, or
    other modalities). It supports rotary positional encoding (rope), optional causal temporal positioning, and can
    substitute padded positions with learnable registers. The module is highly configurable for head size, number of
    layers, and register usage.
    Args:
        attention_head_dim (int): Dimension of each attention head (default=128).
        num_attention_heads (int): Number of attention heads (default=30).
        num_layers (int): Number of transformer layers (default=2).
        positional_embedding_theta (float): Scaling factor for position embedding (default=10000.0).
        positional_embedding_max_pos (list[int] | None): Max positions for positional embeddings (default=[1]).
        causal_temporal_positioning (bool): If True, uses causal attention (default=False).
        num_learnable_registers (int | None): Number of learnable registers to replace padded tokens. If None, disables
            register replacement. (default=128)
        rope_type (LTXRopeType): The RoPE variant to use (default=DEFAULT_ROPE_TYPE).
        double_precision_rope (bool): Use double precision rope calculation (default=False).
        register_padding_mode (str): Register replacement mode. "legacy" preserves the original behavior, while
            "batch_safe" supports batched right-padded inputs. (default="legacy")
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        register_padding_mode: str = "legacy",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        if register_padding_mode not in {"legacy", "batch_safe"}:
            raise ValueError(
                f"Unsupported register_padding_mode: {register_padding_mode}. Expected one of: 'legacy', 'batch_safe'."
            )
        self.register_padding_mode = register_padding_mode
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.register_padding_mode == "batch_safe":
            return self._replace_padded_with_learnable_registers_batch_safe(hidden_states, attention_mask)
        return self._replace_padded_with_learnable_registers_legacy(hidden_states, attention_mask)

    def _replace_padded_with_learnable_registers_legacy(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )

        num_registers_duplications = hidden_states.shape[1] // self.num_learnable_registers
        learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()

        non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_nums = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_nums
        adjusted_hidden_states = torch.nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * learnable_registers

        attention_mask = torch.full_like(
            attention_mask,
            0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def _replace_padded_with_learnable_registers_batch_safe(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {seq_len} must be divisible by num_learnable_registers {self.num_learnable_registers}."
        )

        learnable_registers = self.learnable_registers.repeat(seq_len // self.num_learnable_registers, 1)
        learnable_registers = learnable_registers.to(device=hidden_states.device, dtype=hidden_states.dtype)
        learnable_registers = learnable_registers.unsqueeze(0).expand(batch_size, -1, -1)

        valid_mask = attention_mask[:, 0, 0, :] >= 0
        token_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        valid_counts = valid_mask.sum(dim=1)

        right_padding_mask = token_indices < valid_counts.unsqueeze(1)
        is_right_padded = torch.all(valid_mask == right_padding_mask, dim=1)

        right_padded_states = torch.where(valid_mask.unsqueeze(-1), hidden_states, learnable_registers)

        compact_order = torch.argsort(valid_mask.to(torch.int32), dim=1, descending=True, stable=True)
        compact_states = torch.gather(hidden_states, 1, compact_order.unsqueeze(-1).expand_as(hidden_states))
        compact_valid_mask = token_indices < valid_counts.unsqueeze(1)
        legacy_left_padded_states = torch.where(compact_valid_mask.unsqueeze(-1), compact_states, learnable_registers)

        hidden_states = torch.where(is_right_padded.view(batch_size, 1, 1), right_padded_states, legacy_left_padded_states)
        attention_mask = torch.zeros_like(attention_mask)

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Embeddings1DConnector.
        Args:
            hidden_states (torch.Tensor): Input tensor of embeddings (shape [batch, seq_len, feature_dim]).
            attention_mask (torch.Tensor|None): Optional mask for valid tokens (shape compatible with hidden_states).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed features and the corresponding (possibly modified) mask.
        """
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
        if self.register_padding_mode == "batch_safe":
            indices_grid = indices_grid.expand(hidden_states.shape[0], -1, -1)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


class Embeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    @classmethod
    def from_config(cls: type[Embeddings1DConnector], config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
        num_attention_heads = transformer_config.get("connector_num_attention_heads", 30)
        attention_head_dim = transformer_config.get("connector_attention_head_dim", 128)
        num_layers = transformer_config.get("connector_num_layers", 2)

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
            register_padding_mode=transformer_config.get("connector_register_padding_mode", "legacy"),
        )
        return connector


class AudioEmbeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    @classmethod
    def from_config(cls: type[Embeddings1DConnector], config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
        num_attention_heads = transformer_config.get(
            "audio_connector_num_attention_heads", transformer_config.get("connector_num_attention_heads", 30)
        )
        attention_head_dim = transformer_config.get(
            "audio_connector_attention_head_dim", transformer_config.get("connector_attention_head_dim", 128)
        )
        num_layers = transformer_config.get("audio_connector_num_layers", transformer_config.get("connector_num_layers", 2))

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
            register_padding_mode=transformer_config.get(
                "audio_connector_register_padding_mode",
                transformer_config.get("connector_register_padding_mode", "legacy"),
            ),
        )
        return connector
