"""
StitchImage2D: Stitch a diffusion VAE + DiT prefix into a reward-model encoder head.

Architecture:
    VAE latent -> [noise added externally] -> DiT blocks[:M] ->
    Conv2d stitching -> bilinear interpolation to the head's patch grid ->
    reward-encoder blocks[N:] ->
    (image_embeds, hidden_states) in training / (B, D) pooled at inference

Operates on 2D images (SD3.5-Medium / SD3.5-Large / FLUX backbones, stitched
into a pretrained reward-model encoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from stitchvm.models.feedforward.clip_head import CLIPEncoderHead
from stitchvm.models.backbones.flux import FluxDiTStitched
from stitchvm.models.stitching_layer_builder import ResidualStitchingLayer


class StitchImage2D(nn.Module):
    def __init__(
        self,
        diffusion_vae: nn.Module,
        clip_head: CLIPEncoderHead,
        stitching_layer_config,
        stitching_layer_init_path: str = None,
        device: torch.device = None,
        dit: nn.Module = None,
    ):
        super().__init__()
        if dit is None:
            raise ValueError(
                "StitchImage2D requires a DiT prefix (dit=...). Every supported "
                "stitching location has the form enc_blocks_N_dit_M."
            )
        self.device = device
        self.diffusion_vae = diffusion_vae
        # Named stitched_3d_model for checkpoint save/load compatibility
        self.stitched_3d_model = clip_head
        self.stitched_dit = dit

        # Reward-head patch grid (target of the post-stitch interpolation)
        self.clip_patches_h = clip_head.num_patches_h
        self.clip_patches_w = clip_head.num_patches_w

        # Build stitching layer
        in_channels = dit.inner_dim
        stitching_conv = stitching_layer_config.build(in_channels=in_channels)

        # Validate output channels match feedforward head's hidden size
        expected_out = clip_head.hidden_size
        actual_out = stitching_conv.out_channels
        if actual_out != expected_out:
            raise ValueError(
                f"Stitching layer output channels ({actual_out}) must match "
                f"feedforward head hidden size ({expected_out}). "
                f"Use 'o{expected_out}' in the conv spec."
            )

        if stitching_layer_init_path is not None:
            logger.info(f"Loading stitching layer from {stitching_layer_init_path}")
            state_dict = torch.load(
                stitching_layer_init_path, map_location=device, weights_only=True
            )
            stitching_conv.load_state_dict(state_dict, strict=True)

        self.stitching_layer = ResidualStitchingLayer(stitching_conv)

    @torch.no_grad()
    def vae_encoder_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to a normalized VAE latent.

        Args:
            images: (B, C, H, W) in [-1, 1]

        Returns:
            (B, latent_channels, H/8, W/8) normalized latent
        """
        latents = self.diffusion_vae.encode(images).latent_dist.mode()
        shift_factor = getattr(self.diffusion_vae.config, "shift_factor", 0.0)
        scaling_factor = getattr(self.diffusion_vae.config, "scaling_factor", 1.0)
        return (latents - shift_factor) * scaling_factor

    def forward(
        self,
        noisy_latent=None,
        timestep=None,
        encoder_hidden_states=None,
        pooled_projections=None,
        train=False,
    ):
        """
        Args:
            noisy_latent: (B, C, H', W') pre-computed noisy VAE latent
                (encode images beforehand with vae_encoder_forward)
            timestep: (B,) diffusion timestep = noise sigma * 1000 for every backbone
            encoder_hidden_states: text embeddings for the DiT
            pooled_projections: pooled text embeddings for the DiT
            train: if True, return the (image_embeds, hidden_states) tuple the
                distillation losses consume (hidden_states: (B, 1+N, D) incl.
                CLS, pre-post_layernorm); else return (B, D) pooled embedding.

        Returns:
            (image_embeds, hidden_states) when train=True,
            (B, D) pooled embedding otherwise.
        """
        if noisy_latent is None:
            raise ValueError("noisy_latent is required (see vae_encoder_forward)")
        latent = noisy_latent

        # 1. DiT feature extraction.
        # Cast all inputs to DiT dtype (e.g. bfloat16) to avoid dtype mismatch
        dit_dtype = next(self.stitched_dit.parameters()).dtype
        # The public stitched-model API uses [0, 1000]; FLUX's prefix wrapper
        # explicitly accepts raw sigma, including values close to zero.
        dit_timestep = timestep / 1000 if isinstance(self.stitched_dit, FluxDiTStitched) else timestep
        features = self.stitched_dit(
            hidden_states=latent.to(dit_dtype),
            timestep=dit_timestep,
            encoder_hidden_states=encoder_hidden_states.to(dit_dtype) if encoder_hidden_states is not None else None,
            pooled_projections=pooled_projections.to(dit_dtype) if pooled_projections is not None else None,
        )

        # 2. Stitching layer + bilinear interpolation to the head's patch grid
        # (a downsample at the evaluation resolutions: DiT grids are larger than
        # the 16/24/27-token head grids). Match features dtype to the stitching
        # layer's weight dtype.
        target = (self.clip_patches_h, self.clip_patches_w)
        stitching_dtype = next(self.stitching_layer.parameters()).dtype
        stitched = self.stitching_layer(features.to(stitching_dtype))
        stitched = F.interpolate(
            stitched, size=target, mode="bilinear", align_corners=True
        )

        # 3. Reward-encoder head
        output = self.stitched_3d_model(stitched, return_patch_tokens=train)
        return output
