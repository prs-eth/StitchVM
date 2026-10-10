"""Frozen FLUX.1-dev feature backbone for Stage 1.

Packs spatial VAE latents into 2x2 tokens, applies timestep and guidance
conditioning, and returns a spatial feature map. Stage 1 retains the first
half of the combined dual-stream and single-stream block sequence.
"""

from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def _prepare_latent_image_ids(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build Flux image-token rotary ids for a (height, width) packed grid.

    Inlined from ``diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids``
    (static method in diffusers 0.37). Returns a ``(height * width, 3)``
    tensor whose columns are ``(0, row_index, col_index)``. We avoid
    importing the pipeline module to keep this backbone self-contained.
    """
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    ids = ids.reshape(height * width, 3)
    return ids.to(device=device, dtype=dtype)


class FluxDiTStitched(nn.Module):
    """Wraps a :class:`FluxTransformer2DModel`, runs only the first
    ``split_layer`` blocks and returns a spatial feature map.

    ``split_layer`` is counted linearly across both block types:
      - ``split_layer <= 19``: keep ``transformer_blocks[:split_layer]``
        and no single blocks.
      - ``split_layer > 19``: keep all 19 ``transformer_blocks`` and
        ``single_transformer_blocks[:split_layer - 19]``.

    The forward method accepts a **spatial** latent ``(B, 16, H/8, W/8)``
    and handles Flux's 2x2 packing internally. Output shape is
    ``(B, inner_dim=3072, H/16, W/16)``.

    Unused blocks, ``norm_out`` and ``proj_out`` are dropped. All kept
    parameters are frozen (the DiT prefix is never trained).
    """

    def __init__(
        self,
        dit,  # FluxTransformer2DModel
        split_layer: int,
        guidance_scale: float = 3.5,
    ):
        super().__init__()
        self.split_layer = split_layer
        self.guidance_scale = float(guidance_scale)
        self.dit_config = dit.config

        total_joint = len(dit.transformer_blocks)
        total_single = len(dit.single_transformer_blocks)
        total = total_joint + total_single
        assert 0 < split_layer <= total, (
            f"split_layer must be in [1, {total}], got {split_layer}"
        )

        # Keep shared components (attribute names match Flux's own, NOT
        # SD3's — Flux uses ``x_embedder`` for the packed-patch embedder
        # and ``pos_embed`` for the rotary module).
        self.x_embedder = dit.x_embedder
        self.context_embedder = dit.context_embedder
        self.time_text_embed = dit.time_text_embed
        self.pos_embed = dit.pos_embed

        # Partition blocks
        if split_layer <= total_joint:
            self.transformer_blocks = nn.ModuleList(
                list(dit.transformer_blocks[:split_layer])
            )
            self.single_transformer_blocks = nn.ModuleList()
        else:
            self.transformer_blocks = nn.ModuleList(list(dit.transformer_blocks))
            single_needed = split_layer - total_joint
            self.single_transformer_blocks = nn.ModuleList(
                list(dit.single_transformer_blocks[:single_needed])
            )

        # Drop the unused tail so refcount can free it
        del dit

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        logger.info(
            f"FluxDiTStitched: using first {split_layer}/{total} blocks "
            f"({len(self.transformer_blocks)} dual + "
            f"{len(self.single_transformer_blocks)} single). "
            f"guidance_scale={self.guidance_scale}. "
            f"Removed {total - split_layer} unused blocks."
        )

    @property
    def inner_dim(self) -> int:
        return self.dit_config.num_attention_heads * self.dit_config.attention_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run the first ``split_layer`` blocks of the Flux DiT.

        Args:
            hidden_states: ``(B, 16, H_lat, W_lat)`` spatial VAE latent,
                where ``H_lat, W_lat`` must both be even (Flux requires
                2x2 packing). Typically ``H_lat = H/8`` from the VAE.
            timestep: ``(B,)`` noise sigma in [0, 1]. The wrapper multiplies by 1000 to
                match ``FluxTransformer2DModel.forward``.
            encoder_hidden_states: ``(B, 512, 4096)`` T5 embeddings.
            pooled_projections: ``(B, 768)`` CLIP pooled embeddings.
            joint_attention_kwargs: forwarded to each block, same as
                diffusers' own forward. Pass ``None`` unless you know you
                need it.

        Returns:
            ``(B, 3072, H_lat/2, W_lat/2)`` spatial feature map, ready to
            feed into a Conv2d stitching layer. Equivalent spatial
            resolution to SD3 with ``patch_size=2``.
        """
        # --- 0. Shape bookkeeping ---------------------------------------
        B, C, H_lat, W_lat = hidden_states.shape
        assert H_lat % 2 == 0 and W_lat % 2 == 0, (
            f"Flux requires even latent spatial dims for 2x2 packing; "
            f"got H_lat={H_lat}, W_lat={W_lat}"
        )
        H_pack, W_pack = H_lat // 2, W_lat // 2
        device = hidden_states.device
        dtype = hidden_states.dtype

        # --- 1. Pack 2x2 patches -----------------------------------------
        # Mirrors ``FluxPipeline._pack_latents`` (pipeline_flux.py:521-526).
        # Token t = i*W_pack + j holds the 2x2 block at (i, j) — strict
        # row-major ordering over (H_pack, W_pack).
        hidden_states = hidden_states.view(B, C, H_pack, 2, W_pack, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(B, H_pack * W_pack, C * 4)

        # --- 2. Patch embedding (Linear 64 -> 3072) ----------------------
        hidden_states = self.x_embedder(hidden_states)

        # --- 3. Timestep + guidance + pooled embedding -------------------
        # Both timestep and guidance are multiplied by 1000 inside
        # FluxTransformer2DModel.forward (transformer_flux.py:682,684).
        #
        # Stage 1 supplies raw sigma in [0, 1]. Scale it by 1000 to
        # match FluxTransformer2DModel. Callers must pass raw sigma.
        timestep = timestep.to(dtype) * 1000
        guidance = torch.full(
            (B,), self.guidance_scale, device=device, dtype=dtype
        ) * 1000
        temb = self.time_text_embed(timestep, guidance, pooled_projections)

        # --- 4. Text embedding (Linear 4096 -> 3072) --------------------
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # --- 5. Build rotary ids + pos_embed -----------------------------
        # Mirrors transformer_flux.py:693-707. txt_ids is all-zeros so
        # the text tokens all land at rotary position 0 — standard Flux
        # behavior, matching pipeline_flux.py:386.
        txt_ids = torch.zeros(
            encoder_hidden_states.shape[1], 3, device=device, dtype=dtype
        )
        img_ids = _prepare_latent_image_ids(H_pack, W_pack, device, dtype)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)  # tuple (cos, sin)

        # --- 6. Dual-stream blocks ---------------------------------------
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # --- 7. Single-stream blocks -------------------------------------
        # NOTE: in diffusers 0.37 ``FluxSingleTransformerBlock.forward``
        # takes both tensors separately, concatenates them internally,
        # splits them back, and returns ``(encoder, hidden)``. So we pass
        # them the same way as the dual-stream loop — NO manual concat at
        # the boundary.
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # --- 8. Unpack back to spatial -----------------------------------
        # ``hidden_states`` has shape (B, L=H_pack*W_pack, 3072). A plain
        # ``transpose + reshape`` recovers the (H_pack, W_pack) row-major
        # grid: output[b, c, i, j] == input[b, i*W_pack + j, c] ==
        # the feature vector for the 2x2 block at (i, j). Do NOT use
        # ``_unpack_latents`` from the Flux pipeline here — that function
        # interprets the channel dim as encoding 2x2 sub-pixels, which is
        # only valid for the 64-channel raw packed latent, not for the
        # 3072-channel DiT embedding.
        hidden_states = hidden_states.transpose(1, 2).reshape(
            B, -1, H_pack, W_pack
        )

        return hidden_states
