import copy

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.checkpoint import checkpoint


class CLIPEncoderHead(nn.Module):
    """CLIP ViT encoder (latter blocks) as feedforward head for stitching.

    For layer finding: register hooks on the full CLIP encoder to extract
    intermediate features at each block.

    For stitching: takes spatially-formatted features from the stitching layer,
    converts them to patch tokens, runs through the remaining encoder blocks,
    and produces a CLIP image embedding.

    Pipeline:
        stitched features (B, C, H', W')
        -> flatten to patch tokens (B, N, C)
        -> remaining encoder blocks
        -> layer norm -> pool -> image embedding (B, D)
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        post_layernorm: nn.LayerNorm,
        visual_projection: nn.Linear,
        class_embedding: nn.Parameter,
        position_embedding: nn.Embedding,
        hidden_size: int,
        num_patches_h: int,
        num_patches_w: int,
    ):
        super().__init__()
        self.layers = encoder_layers
        self.post_layernorm = post_layernorm
        self.visual_projection = visual_projection
        self.class_embedding = class_embedding
        self.position_embedding = position_embedding
        self.hidden_size = hidden_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.gradient_checkpointing = False

        num_positions = num_patches_h * num_patches_w + 1  # CLS + patches
        self.register_buffer("position_ids", torch.arange(num_positions).unsqueeze(0))

        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_clip_vision_model(
        cls,
        vision_model,
        split_layer: int,
    ) -> "CLIPEncoderHead":
        """Create a CLIPEncoderHead by splitting a CLIPVisionModelWithProjection.

        Args:
            vision_model: transformers.CLIPVisionModelWithProjection (the full model)
            split_layer: number of encoder layers to discard (kept by backbone)
        """
        config = vision_model.config
        if hasattr(config, "vision_config"):
            config = config.vision_config

        total_layers = len(vision_model.vision_model.encoder.layers)
        assert 0 < split_layer <= total_layers, (
            f"split_layer must be in [1, {total_layers}], got {split_layer}"
        )

        # Deep copy to avoid sharing modules with the teacher model.
        # Without this, fine-tuning the head would update the teacher's own
        # layers in place, making the distillation target drift during training.
        kept_layers = copy.deepcopy(
            nn.ModuleList(list(vision_model.vision_model.encoder.layers[split_layer:]))
        )
        post_ln = copy.deepcopy(vision_model.vision_model.post_layernorm)
        vis_proj = copy.deepcopy(vision_model.visual_projection)
        class_emb = copy.deepcopy(vision_model.vision_model.embeddings.class_embedding)
        pos_emb = copy.deepcopy(vision_model.vision_model.embeddings.position_embedding)

        image_size = config.image_size
        patch_size = config.patch_size
        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size

        head = cls(
            encoder_layers=kept_layers,
            post_layernorm=post_ln,
            visual_projection=vis_proj,
            class_embedding=class_emb,
            position_embedding=pos_emb,
            hidden_size=config.hidden_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )

        logger.info(
            f"CLIPEncoderHead: keeping layers [{split_layer}..{total_layers}), "
            f"discarded first {split_layer} layers. "
            f"hidden_size={config.hidden_size}, "
            f"patches=({num_patches_h}, {num_patches_w})"
        )
        return head

    def forward(
        self,
        stitched_features: torch.Tensor,
        return_patch_tokens: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run remaining CLIP encoder blocks on stitched features.

        Args:
            stitched_features: (B, C, H', W') spatial features
            return_patch_tokens: if True, additionally return the final
                (B, N+1, D) hidden states (CLS + patches, pre-post_layernorm)
                for dense distillation — the same tensor the teacher exposes
                as last_hidden_state.

        Returns:
            (image_embeds, hidden_states) tuple when return_patch_tokens=True,
            (B, D) pooled CLIP image embedding otherwise.
        """
        B, C, H, W = stitched_features.shape
        # Reshape to sequence: (B, N, C) where N = H * W
        hidden_states = stitched_features.flatten(2).transpose(1, 2)

        # Prepend CLS token (matching original CLIP encoder behavior)
        cls_tokens = self.class_embedding.expand(B, 1, -1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        # Add position embeddings (CLS at idx 0, patches at idx 1..N)
        hidden_states = hidden_states + self.position_embedding(self.position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and hidden_states.requires_grad:
                hidden_states = checkpoint(
                    layer, hidden_states, None, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=None)

        # Pool from CLS token + post_layernorm + projection (matching original CLIP)
        pooled = hidden_states[:, 0, :]
        pooled = self.post_layernorm(pooled)
        image_embeds = self.visual_projection(pooled)

        if return_patch_tokens:
            return image_embeds, hidden_states

        return image_embeds


def register_clip_feature_hooks(
    clip_vision_model,
    feature_store: dict[str, list],
    num_blocks: int | None = None,
) -> list:
    """Register forward hooks on the first `num_blocks` CLIP encoder layers.

    Captures patch tokens (excluding CLS) at each layer, reshaped to spatial
    format (C, H', W') per sample.

    Args:
        clip_vision_model: transformers.CLIPVisionModel
        feature_store: dict to accumulate features, keys will be "enc_blocks_{i}"
        num_blocks: number of leading encoder layers to hook (None = all)

    Returns:
        list of hook handles (call handle.remove() when done)
    """

    def make_hook(name):
        def hook(module, input, output):
            # CLIPEncoderLayer returns a tuple (hidden_states,) or
            # (hidden_states, attn_weights); extract the tensor first.
            hidden = output[0] if isinstance(output, tuple) else output
            # Exclude CLS token: (B, N, D)
            patches = hidden[:, 1:, :]
            # Reshape to spatial: (B, D, H', W'). Stage 1 always feeds CLIP
            # its native square input size, so N is a perfect square and
            # H' = W' = sqrt(N).
            inp = input[0] if isinstance(input, tuple) else input
            B_in, N_plus_cls, _ = inp.shape
            N = N_plus_cls - 1  # exclude CLS
            h_patches = int(round(N**0.5))
            w_patches = N // h_patches
            B, _, D = patches.shape
            spatial = patches.transpose(1, 2).reshape(B, D, h_patches, w_patches)
            # Store as fp16 per sample
            for i in range(B):
                if name not in feature_store:
                    feature_store[name] = []
                feature_store[name].append(
                    spatial[i].detach().cpu().to(dtype=torch.float16)
                )
            del spatial, patches, hidden

        return hook

    layers = clip_vision_model.vision_model.encoder.layers
    if num_blocks is not None:
        layers = layers[:num_blocks]

    handles = []
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(f"enc_blocks_{idx + 1}"))
        handles.append(handle)

    return handles
