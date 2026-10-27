"""Qwen2-VL vision encoder head (HPSv3) for stitching.

HPSv3 is Qwen2-VL-7B-Instruct with a reward head. Its 675M Qwen2-VL vision
tower has a Conv3d patch embedding (2 x 14 x 14), 32 blocks with 2D rotary
position embeddings, hidden size 1280, and a 2x2 PatchMerger projecting to the
language-model width (3584). Only the vision branch is stitched; the language
branch and reward head are untouched and consume the merged tokens exactly as
in the original model.

Token order. The Qwen2-VL processor flattens each image into 2x2 merge windows,
(t, h/2, w/2, 2, 2). The head converts spatial features to that order before
the blocks, so the rotary embeddings, the per-image attention segments
(cu_seqlens) and the PatchMerger see exactly what the native tower sees after
its first N blocks.

For stitching-layer search, register hooks on visual.blocks.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.checkpoint import checkpoint

# Qwen2-VL image processor statistics (preprocessor_config.json of
# Qwen2-VL-7B-Instruct): the OpenAI CLIP mean / std.
QWEN2VL_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN2VL_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class Qwen2VLProcessorStats:
    """Minimal processor providing image_mean/std for preprocessing."""

    image_mean = list(QWEN2VL_IMAGE_MEAN)
    image_std = list(QWEN2VL_IMAGE_STD)


def spatial_to_qwen_tokens(x: torch.Tensor, merge: int = 2) -> torch.Tensor:
    """(B, C, t, h, w) spatial features -> (B, t*h*w, C) in Qwen2-VL token order."""
    B, C, t, h, w = x.shape
    x = x.reshape(B, C, t, h // merge, merge, w // merge, merge)
    x = x.permute(0, 2, 3, 5, 4, 6, 1)  # (B, t, h/m, w/m, m, m, C)
    return x.reshape(B, t * h * w, C)


def qwen_tokens_to_spatial(
    tokens: torch.Tensor, t: int, h: int, w: int, merge: int = 2
) -> torch.Tensor:
    """Inverse of spatial_to_qwen_tokens: (B, t*h*w, C) -> (B, C, t, h, w)."""
    B, _N, C = tokens.shape
    x = tokens.reshape(B, t, h // merge, w // merge, merge, merge, C)
    x = x.permute(0, 6, 1, 2, 4, 3, 5)  # (B, C, t, h/m, m, w/m, m)
    return x.reshape(B, C, t, h, w)


def qwen2vl_patchify(
    frames: torch.Tensor,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    mean=QWEN2VL_IMAGE_MEAN,
    std=QWEN2VL_IMAGE_STD,
):
    """Torch port of Qwen2VLImageProcessor._preprocess after resizing.

    Args:
        frames: (B, 3, H, W) images in [0, 1] (or (B, T, 3, H, W)); an image is
            repeated to temporal_patch_size frames, as the processor does. H, W
            must be multiples of patch_size * merge_size.

    Returns:
        pixel_values: (B * gt * gh * gw, 3 * tp * p * p)
        grid_thw: (B, 3) long tensor of (gt, gh, gw)
    """
    if frames.ndim == 4:
        frames = frames.unsqueeze(1)
    B, T, C, H, W = frames.shape
    mean_t = torch.tensor(mean, device=frames.device, dtype=frames.dtype).view(1, 1, C, 1, 1)
    std_t = torch.tensor(std, device=frames.device, dtype=frames.dtype).view(1, 1, C, 1, 1)
    x = (frames - mean_t) / std_t
    if T % temporal_patch_size != 0:
        # processor: np.tile of the last frame up to a multiple of tp
        pad = temporal_patch_size - T % temporal_patch_size
        x = torch.cat([x, x[:, -1:].expand(B, pad, C, H, W)], dim=1)
        T = T + pad
    gt, gh, gw = T // temporal_patch_size, H // patch_size, W // patch_size
    x = x.reshape(
        B, gt, temporal_patch_size, C,
        gh // merge_size, merge_size, patch_size,
        gw // merge_size, merge_size, patch_size,
    )
    # (B, gt, gh/m, gw/m, m, m, C, tp, p, p) — Qwen2VLImageProcessor transpose
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    pixel_values = x.reshape(B * gt * gh * gw, C * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([[gt, gh, gw]] * B, device=frames.device, dtype=torch.long)
    return pixel_values, grid_thw


def _rot_pos_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """Same position ids as Qwen2VisionTransformerPretrainedModel.rot_pos_emb."""
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(pos_ids, dim=0)


def qwen2vl_position_embeddings(rotary_pos_emb: nn.Module, grid_thw: torch.Tensor, merge: int):
    """(cos, sin) of the vision 2D rotary embedding for packed tokens."""
    pos_ids = _rot_pos_ids(grid_thw.cpu(), merge).to(rotary_pos_emb.inv_freq.device)
    max_grid = int(grid_thw[:, 1:].max())
    freqs = rotary_pos_emb(max_grid)[pos_ids].flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def qwen2vl_cu_seqlens(grid_thw: torch.Tensor, device) -> torch.Tensor:
    """Attention segments: one per temporal patch group of each image."""
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    return F.pad(cu, (1, 0), value=0).to(device)


class Qwen2VLVisionHead(nn.Module):
    """Qwen2-VL vision blocks [N:] + PatchMerger as the stitched feedforward head.

    Pipeline:
        stitched features (B, C, h, w)
        -> Qwen2-VL token order (B*h*w, C)
        -> remaining vision blocks (2D RoPE, per-image attention segments)
        -> PatchMerger -> (B, h*w/4, D_llm) tokens for the language branch
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        merger: nn.Module,
        rotary_pos_emb: nn.Module,
        hidden_size: int,
        spatial_merge_size: int,
        num_patches_h: int,
        num_patches_w: int,
    ):
        super().__init__()
        self.layers = blocks
        self.merger = merger
        self.rotary_pos_emb = rotary_pos_emb
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.gradient_checkpointing = False

        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_qwen2vl_visual(
        cls,
        visual: nn.Module,
        split_layer: int,
        num_patches_h: int = 32,
        num_patches_w: int = 32,
        attn_implementation: str = "sdpa",
    ) -> "Qwen2VLVisionHead":
        """Create by splitting a Qwen2-VL vision tower at split_layer.

        Args:
            visual: Qwen2VisionTransformerPretrainedModel
            split_layer: number of vision blocks to discard
            num_patches_h/w: default patch grid (14 px patches) used as the
                post-stitch interpolation target. 32 x 32 is the grid of the
                448 x 448 training input. Evaluation passes each image's
                native HPSv3 grid instead (StitchImage2D target_grid).
            attn_implementation: attention kernel of the copied (trainable,
                fp32) blocks; "sdpa" works in fp32 and under bf16 autocast,
                unlike flash_attention_2. The source tower is not affected.
        """
        total = len(visual.blocks)
        assert 0 < split_layer <= total, f"split_layer must be in [1, {total}], got {split_layer}"
        kept = copy.deepcopy(nn.ModuleList(list(visual.blocks[split_layer:])))
        for block in kept:
            block.attn.config = copy.copy(block.attn.config)
            block.attn.config._attn_implementation = attn_implementation
        merger = copy.deepcopy(visual.merger)
        rotary = copy.deepcopy(visual.rotary_pos_emb)
        hidden_size = visual.patch_embed.embed_dim
        head = cls(
            blocks=kept,
            merger=merger,
            rotary_pos_emb=rotary,
            hidden_size=hidden_size,
            spatial_merge_size=visual.spatial_merge_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        logger.info(
            f"Qwen2VLVisionHead: keeping blocks [{split_layer}..{total}), discarded first "
            f"{split_layer}. hidden_size={hidden_size}, merger_out={merger.mlp[-1].out_features}, "
            f"default patches=({num_patches_h}, {num_patches_w})"
        )
        return head

    def forward(
        self,
        stitched_features: torch.Tensor,
        return_patch_tokens: bool = False,
        **kwargs,
    ):
        """Run the remaining Qwen2-VL vision blocks on stitched features.

        Args:
            stitched_features: (B, C, h, w) features on the 14-px patch grid
                (h, w even).
            return_patch_tokens: if True, return (merged_tokens, hidden_states)

        Returns:
            merged tokens (B, h*w/4, D_llm), or the tuple
            (merged tokens, hidden_states (B, h*w, C)) — hidden_states are the
            last block's output before the merger, in Qwen2-VL token order.
        """
        x = stitched_features
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, t, h, w = x.shape
        m = self.spatial_merge_size
        hidden = spatial_to_qwen_tokens(x, m).reshape(B * t * h * w, C)

        grid_thw = torch.tensor([[t, h, w]] * B, dtype=torch.long)
        position_embeddings = qwen2vl_position_embeddings(self.rotary_pos_emb, grid_thw, m)
        cu_seqlens = qwen2vl_cu_seqlens(grid_thw, hidden.device)

        for block in self.layers:
            if self.gradient_checkpointing and hidden.requires_grad:
                hidden = checkpoint(
                    block, hidden, cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings, use_reentrant=False,
                )
            else:
                hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        merged = self.merger(hidden)
        merged = merged.reshape(B, t * h * w // (m * m), -1)
        if return_patch_tokens:
            return merged, hidden.reshape(B, t * h * w, C)
        return merged


class Qwen2VLTeacherWrapper(nn.Module):
    """Full Qwen2-VL vision tower as the distillation teacher.

    Takes images in [0, 1] (B, 3, H, W), applies the native Qwen2-VL
    normalization + patchification, and returns an output with .image_embeds
    (merged tokens, B x N/4 x D_llm) and .last_hidden_state (last block output
    before the merger, B x N x 1280).
    """

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values=None, frames=None, **kwargs):
        if frames is None:
            raise ValueError("Qwen2VLTeacherWrapper needs images in [0, 1] (frames=...)")
        B = frames.shape[0]
        flat, grid_thw = qwen2vl_patchify(frames.float())
        dtype = self.visual.patch_embed.proj.weight.dtype
        out = self.visual(flat.to(dtype), grid_thw=grid_thw)
        hidden = out.last_hidden_state
        merged = out.pooler_output
        return _Qwen2VLOutput(
            image_embeds=merged.reshape(B, merged.shape[0] // B, -1),
            last_hidden_state=hidden.reshape(B, hidden.shape[0] // B, -1),
        )


class _Qwen2VLOutput:
    """Simple output container matching CLIPVisionModelOutput interface."""

    __slots__ = ("image_embeds", "last_hidden_state")

    def __init__(self, image_embeds, last_hidden_state):
        self.image_embeds = image_embeds
        self.last_hidden_state = last_hidden_state


def register_qwen2vl_feature_hooks(
    visual: nn.Module,
    feature_store: dict[str, list],
    grid: dict,
    num_blocks: int | None = None,
) -> list:
    """Register forward hooks on the first `num_blocks` Qwen2-VL vision blocks.

    Captures every block output, converts it from Qwen2-VL token order back to
    spatial format and stores one (C, h, w) tensor per image. `grid` must hold
    the current {"B", "t", "h", "w"} of the batch (set by the caller before
    each forward). Used for the HPSv3 and Qwen3.5 towers alike.
    """
    merge = visual.spatial_merge_size

    def make_hook(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            B, t, h, w = grid["B"], grid["t"], grid["h"], grid["w"]
            spatial = qwen_tokens_to_spatial(out.reshape(B, t * h * w, -1), t, h, w, merge)[:, :, 0]
            for i in range(B):
                feature_store.setdefault(name, []).append(
                    spatial[i].detach().cpu().to(dtype=torch.float16)
                )

        return hook

    blocks = visual.blocks if num_blocks is None else visual.blocks[:num_blocks]
    return [b.register_forward_hook(make_hook(f"enc_blocks_{i + 1}")) for i, b in enumerate(blocks)]
