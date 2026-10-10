from typing import Any

import torch
import torch.nn as nn
from loguru import logger


class SD3DiTStitched(nn.Module):
    """Wraps an SD3Transformer2DModel, runs only the first `split_layer` blocks.

    Unused blocks and output head are deleted. Output is reshaped to spatial
    format: (B, inner_dim, H', W').
    """

    def __init__(
        self,
        dit,  # SD3Transformer2DModel
        split_layer: int,
    ):
        super().__init__()
        self.split_layer = split_layer
        self.dit_config = dit.config

        total = len(dit.transformer_blocks)
        assert 0 < split_layer <= total, (
            f"split_layer must be in [1, {total}], got {split_layer}"
        )

        # Keep shared components
        self.pos_embed = dit.pos_embed
        self.time_text_embed = dit.time_text_embed
        self.context_embedder = dit.context_embedder

        self.joint_blocks = nn.ModuleList(list(dit.transformer_blocks[:split_layer]))

        del dit

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        logger.info(
            f"SD3DiTStitched: using first {split_layer}/{total} blocks. "
            f"Removed {total - split_layer} unused blocks."
        )

    @property
    def inner_dim(self):
        return self.dit_config.num_attention_heads * self.dit_config.attention_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,  # (B,) float, in [0, 1000]
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # Matches SD3Transformer2DModel.forward exactly (minus output head).
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.joint_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Reshape to spatial: (B, N, D) -> (B, D, H', W')
        p = self.dit_config.patch_size
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0], -1, height // p, width // p
        )

        return hidden_states
