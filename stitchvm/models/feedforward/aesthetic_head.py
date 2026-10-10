"""Aesthetic scorer head: CLIPEncoderHead + frozen MLP.

The MLP predicts a scalar aesthetic score from normalized CLIP image embeddings.
Architecture matches christophschuhmann/improved-aesthetic-predictor:
    768 -> 1024 -> 128 -> 64 -> 16 -> 1

For stitching layer search, the backbone is identical to CLIP — reuse
register_clip_feature_hooks from clip_head.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from stitchvm.models.feedforward.clip_head import CLIPEncoderHead


class AestheticMLP(nn.Module):
    """Aesthetic score MLP (improved-aesthetic-predictor)."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class AestheticEncoderHead(nn.Module):
    """CLIP encoder head + frozen aesthetic MLP.

    Wraps CLIPEncoderHead. The forward() interface is identical to
    CLIPEncoderHead so it is a drop-in replacement in StitchImage2D.
    Use compute_aesthetic_score() to get the scalar score.
    """

    def __init__(self, clip_head: CLIPEncoderHead, aesthetic_mlp: AestheticMLP):
        super().__init__()
        self.clip_head = clip_head
        self.aesthetic_mlp = aesthetic_mlp
        self.aesthetic_mlp.eval()
        for param in self.aesthetic_mlp.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        # The MLP has live Dropout layers but is a frozen scorer: keep it in
        # eval mode even when an ancestor module calls .train(), so both the
        # student score and the teacher target stay deterministic.
        super().train(mode)
        self.aesthetic_mlp.eval()
        return self

    @classmethod
    def from_clip_vision_model(
        cls,
        vision_model,
        split_layer: int,
        mlp_weights_path: str,
    ) -> "AestheticEncoderHead":
        """Create from a CLIPVisionModelWithProjection + MLP checkpoint.

        Args:
            vision_model: transformers.CLIPVisionModelWithProjection
            split_layer: number of encoder layers to discard
            mlp_weights_path: path to aesthetic MLP weights (.pth)
        """
        clip_head = CLIPEncoderHead.from_clip_vision_model(vision_model, split_layer)
        mlp = AestheticMLP()
        state_dict = torch.load(mlp_weights_path, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        logger.info(f"AestheticEncoderHead: loaded MLP from {mlp_weights_path}")
        return cls(clip_head, mlp)

    def forward(
        self,
        stitched_features: torch.Tensor,
        return_patch_tokens: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Same interface as CLIPEncoderHead.forward()."""
        return self.clip_head(
            stitched_features, return_patch_tokens=return_patch_tokens
        )

    def compute_aesthetic_score(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Compute aesthetic score from image embeddings.

        Args:
            image_embeds: (B, D) CLIP image embeddings (post visual_projection)

        Returns:
            (B,) aesthetic scores
        """
        embed_norm = F.normalize(image_embeds.float(), dim=-1)
        # Match input dtype to the MLP's parameter dtype.
        mlp_dtype = next(self.aesthetic_mlp.parameters()).dtype
        return self.aesthetic_mlp(embed_norm.to(mlp_dtype)).squeeze(-1)

    # ── Proxy CLIPEncoderHead properties for StitchImage2D compatibility ──

    @property
    def num_patches_h(self) -> int:
        return self.clip_head.num_patches_h

    @property
    def num_patches_w(self) -> int:
        return self.clip_head.num_patches_w

    @property
    def hidden_size(self) -> int:
        return self.clip_head.hidden_size

    @property
    def gradient_checkpointing(self):
        return self.clip_head.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.clip_head.gradient_checkpointing = value
