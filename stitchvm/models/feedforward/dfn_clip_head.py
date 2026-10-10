"""DFN CLIP encoder head: open_clip ViT-H-14 (apple/DFN5B-CLIP-ViT-H-14-378).

Apple's Data Filtering Networks CLIP: ViT-H/14 trained on curated 5B pairs.
Architecture: 32 ResidualAttentionBlock, hidden=1280, proj=1024, image_size=378.

Same structure as open_clip ViT-H-14, so the split/forward logic mirrors HPSv2EncoderHead.
"""

import copy

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.checkpoint import checkpoint


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class DFNClipEncoderHead(nn.Module):
    """DFN5B-CLIP ViT-H-14 encoder (latter blocks) as feedforward head for stitching.

    Pipeline:
        stitched features (B, C, H', W')
        -> flatten to patch tokens (B, N, C)
        -> remaining ResidualAttentionBlocks
        -> pool CLS -> ln_post -> @ proj -> (B, 1024)
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        ln_post: nn.LayerNorm,
        proj: nn.Parameter,
        class_embedding: nn.Parameter,
        positional_embedding: nn.Parameter,
        hidden_size: int,
        num_patches_h: int,
        num_patches_w: int,
    ):
        super().__init__()
        self.layers = encoder_layers
        self.ln_post = ln_post
        self.proj = proj
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.hidden_size = hidden_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.gradient_checkpointing = False

        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_dfn_clip_model(
        cls,
        dfn_model,
        split_layer: int,
    ) -> "DFNClipEncoderHead":
        """Create by splitting the open_clip visual model at split_layer.

        Args:
            dfn_model: open_clip model (with .visual attribute)
            split_layer: number of transformer blocks to discard
        """
        visual = dfn_model.visual
        total_layers = len(visual.transformer.resblocks)
        assert 0 < split_layer <= total_layers, (
            f"split_layer must be in [1, {total_layers}], got {split_layer}"
        )

        kept_layers = copy.deepcopy(
            nn.ModuleList(list(visual.transformer.resblocks[split_layer:]))
        )
        ln_post = copy.deepcopy(visual.ln_post)
        proj = nn.Parameter(visual.proj.data.clone())
        class_emb = nn.Parameter(visual.class_embedding.data.clone())
        pos_emb = nn.Parameter(visual.positional_embedding.data.clone())

        image_size = visual.image_size
        if isinstance(image_size, (tuple, list)):
            image_h, image_w = image_size[0], image_size[1]
        else:
            image_h = image_w = image_size
        patch_size = visual.conv1.kernel_size[0]
        num_patches_h = image_h // patch_size
        num_patches_w = image_w // patch_size
        hidden_size = visual.conv1.out_channels

        head = cls(
            encoder_layers=kept_layers,
            ln_post=ln_post,
            proj=proj,
            class_embedding=class_emb,
            positional_embedding=pos_emb,
            hidden_size=hidden_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )

        logger.info(
            f"DFNClipEncoderHead: keeping blocks [{split_layer}..{total_layers}), "
            f"discarded first {split_layer} blocks. "
            f"hidden_size={hidden_size}, proj_dim={proj.shape[1]}, "
            f"patches=({num_patches_h}, {num_patches_w})"
        )
        return head

    def forward(
        self,
        stitched_features: torch.Tensor,
        return_patch_tokens: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run remaining DFN CLIP blocks on stitched features.

        Args:
            stitched_features: (B, C, H', W') spatial features
            return_patch_tokens: if True, return (image_embeds, hidden_states)

        Returns:
            (B, proj_dim) image embedding, or (image_embeds, hidden_states) tuple
        """
        B, C, H, W = stitched_features.shape
        hidden_states = stitched_features.flatten(2).transpose(1, 2)  # (B, N, C)

        # Prepend CLS token
        cls_tokens = self.class_embedding.reshape(1, 1, -1).expand(B, -1, -1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        # Add positional embeddings
        hidden_states = hidden_states + self.positional_embedding.unsqueeze(0)

        # The pinned open_clip's ResidualAttentionBlock (batch_first=True)
        # accepts and returns NLD directly — no permutation needed. (The
        # vendored hpsv2 open_clip ships LND blocks instead; see
        # HPSv2EncoderHead.)
        for block in self.layers:
            if self.gradient_checkpointing and hidden_states.requires_grad:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)

        # Pool CLS token + layer norm + projection
        pooled = hidden_states[:, 0, :]
        pooled = self.ln_post(pooled)
        image_embeds = pooled @ self.proj

        if return_patch_tokens:
            return image_embeds, hidden_states
        return image_embeds


class DFNClipTeacherWrapper(nn.Module):
    """Wraps DFN CLIP visual encoder for teacher distillation.

    Produces output with .image_embeds and .last_hidden_state attributes,
    matching the interface expected by LossCLIPStitched.
    """

    def __init__(self, dfn_model):
        super().__init__()
        self.visual = dfn_model.visual
        self._last_hidden_state = None
        self._hook_handle = self.visual.transformer.resblocks[
            -1
        ].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output):
        # batch_first=True blocks return NLD directly; store as-is.
        self._last_hidden_state = output

    def forward(self, pixel_values=None, **kwargs):
        image_embeds = self.visual(pixel_values)
        return _DFNClipOutput(
            image_embeds=image_embeds,
            last_hidden_state=self._last_hidden_state,
        )


class _DFNClipOutput:
    """Simple output container matching CLIPVisionModelOutput interface."""

    __slots__ = ("image_embeds", "last_hidden_state")

    def __init__(self, image_embeds, last_hidden_state):
        self.image_embeds = image_embeds
        self.last_hidden_state = last_hidden_state


class DFNClipProcessor:
    """Minimal processor providing image_mean/std for preprocessing."""

    image_mean = list(OPENAI_DATASET_MEAN)
    image_std = list(OPENAI_DATASET_STD)


def register_dfn_clip_feature_hooks(
    dfn_model,
    feature_store: dict[str, list],
    num_blocks: int | None = None,
) -> list:
    """Register forward hooks on the first `num_blocks` DFN CLIP visual
    transformer blocks (None = all).

    Captures patch tokens (excluding CLS) at each block, reshaped to
    spatial format (D, H', W') per sample.
    """
    visual = dfn_model.visual
    image_size = visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size[0], image_size[1]
    else:
        image_h = image_w = image_size
    patch_size = visual.conv1.kernel_size[0]
    h_patches = image_h // patch_size
    w_patches = image_w // patch_size

    def make_hook(name):
        def hook(module, input, output):
            # batch_first=True: output is (B, N+1, D) NLD already.
            patches = output[:, 1:, :]  # exclude CLS
            B, N, D = patches.shape
            spatial = patches.transpose(1, 2).reshape(B, D, h_patches, w_patches)
            for i in range(B):
                if name not in feature_store:
                    feature_store[name] = []
                feature_store[name].append(
                    spatial[i].detach().cpu().to(dtype=torch.float16)
                )
            del spatial, patches

        return hook

    resblocks = visual.transformer.resblocks
    if num_blocks is not None:
        resblocks = resblocks[:num_blocks]

    handles = []
    for idx, block in enumerate(resblocks):
        handle = block.register_forward_hook(make_hook(f"enc_blocks_{idx + 1}"))
        handles.append(handle)
    return handles
