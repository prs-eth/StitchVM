"""Qwen3.5 vision encoder head (UnifiedReward-2.0-qwen35) for stitching.

Tested checkpoints: CodeGoat24/UnifiedReward-2.0-qwen35-{4b,9b,27b} at the revisions in
TESTED_CHECKPOINTS. Every shape is read from the checkpoint's config.json /
processor_config.json; there is no model-size branch. Verified against transformers 5.3.0
(models/qwen3_5):

- Processor (Qwen2VLImageProcessor(Fast) with the checkpoint's processor_config.json):
  smart_resize to multiples of patch_size * merge_size (32 px) within
  [shortest_edge, longest_edge] pixels, mean/std 0.5, the frame duplicated to
  temporal_patch_size, patches flattened in the 2x2 merge-window order (t, h/2, w/2, 2, 2)
  — the same permutation as Qwen2-VL, so qwen2vl_patchify / spatial_to_qwen_tokens apply
  with the Qwen3.5 patch size and statistics.
- Vision tower (Qwen3_5VisionModel): Conv3d patch embedding + interpolated learned absolute position embedding (added once, before block 0)
  + 2D RoPE recomputed from grid_thw in every block, full attention per image (cu_seqlens),
  then one PatchMerger (LayerNorm -> 4 consecutive rows = one 2x2 window -> MLP -> out width).
- The language model receives only the merger output (masked_scatter at the image
  placeholder tokens). These checkpoints have deepstack_visual_indexes = [] and no DeepStack
  modules; a DeepStack (Qwen3-VL) tower is rejected here, because it feeds additional visual
  features into the language model and would need extra distillation targets.

The head keeps vision blocks [N:] + merger. For N >= 1 the absolute position embedding is
already part of the teacher features at the split, which the stitching layer is fitted to.
Other model types (e.g. qwen3_5_moe, Qwen3-VL) are refused; other qwen3_5 checkpoints load
with a warning, as they have not been tested.
"""

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils.checkpoint import checkpoint

from stitchvm.models.feedforward.qwen2vl_vision_head import (
    qwen2vl_cu_seqlens,
    qwen2vl_patchify,
    spatial_to_qwen_tokens,
)

# Checkpoints this integration was tested with: Hub revision and config.json SHA-256.
# All three share processor_config.json SHA-256 14932921ca485d45...801385b.
TESTED_CHECKPOINTS = {
    "CodeGoat24/UnifiedReward-2.0-qwen35-4b": {
        "revision": "a1cb85ca3dd2c65ff561cc1d8fd66e6830c5ab30",
        "config.json_sha256": "e95206f48999dbf5f9b769f73d2f68c9e47e875a724a63ec4f534bd4e250fc4c",
    },
    "CodeGoat24/UnifiedReward-2.0-qwen35-9b": {
        "revision": "f01548b009741e12ff9817ed91dba94701ed9579",
        "config.json_sha256": "b34df7530e0f189cf528bcc35542d9543b55c7fcddffc06e5a5eeb8849e8641a",
    },
    "CodeGoat24/UnifiedReward-2.0-qwen35-27b": {
        "revision": "3a5c836f485047a96f45754e08c981ca9e41fa60",
        "config.json_sha256": "74d5b93fcfd2b6faa02f00b611e422641e1457dde3dd736c53c95b28f80bb4f7",
    },
}


@dataclass(frozen=True)
class Qwen35ProcessorPolicy:
    """Image policy of the checkpoint's own processor (processor_config.json)."""

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple
    image_std: tuple
    resample: int
    processor_class: str
    image_processor_type: str

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def resized_hw(self, height: int, width: int) -> tuple[int, int]:
        """(H, W) the official processor resizes an image of (height, width) to."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        return smart_resize(
            height, width, factor=self.factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def grid_hw(self, height: int, width: int) -> tuple[int, int]:
        """Vision patch grid (h, w) of an image of (height, width) after the processor resize."""
        rh, rw = self.resized_hw(height, width)
        return rh // self.patch_size, rw // self.patch_size

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "Qwen35ProcessorPolicy":
        path = os.path.join(checkpoint_dir, "processor_config.json")
        if os.path.exists(path):
            cfg = json.load(open(path))
            ip = cfg.get("image_processor", cfg)
            processor_class = cfg.get("processor_class", "")
        else:
            ip = json.load(open(os.path.join(checkpoint_dir, "preprocessor_config.json")))
            processor_class = ip.get("processor_class", "")
        size = ip.get("size", {})
        return cls(
            patch_size=int(ip["patch_size"]),
            merge_size=int(ip["merge_size"]),
            temporal_patch_size=int(ip["temporal_patch_size"]),
            min_pixels=int(size.get("shortest_edge", ip.get("min_pixels"))),
            max_pixels=int(size.get("longest_edge", ip.get("max_pixels"))),
            image_mean=tuple(float(x) for x in ip["image_mean"]),
            image_std=tuple(float(x) for x in ip["image_std"]),
            resample=int(ip.get("resample", 3)),
            processor_class=processor_class,
            image_processor_type=ip.get("image_processor_type", ""),
        )


def qwen35_prepare(frames: torch.Tensor, policy: Qwen35ProcessorPolicy):
    """Images in [0, 1] (B, 3, H, W) -> (pixel_values, grid_thw) as the official processor.

    The batch shares one size (resolution buckets). At sizes the processor keeps unchanged
    (multiples of 32 inside the pixel budget, e.g. every training bucket) no resampling
    happens; otherwise the batch is resized bicubically (the processor's resample=3) and
    clamped to [0, 1] like the processor's uint8 path.
    """
    _, _, H, W = frames.shape
    rh, rw = policy.resized_hw(H, W)
    if (rh, rw) != (H, W):
        frames = TF.resize(frames, [rh, rw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        frames = frames.clamp(0.0, 1.0)
    return qwen2vl_patchify(
        frames.float(),
        patch_size=policy.patch_size,
        temporal_patch_size=policy.temporal_patch_size,
        merge_size=policy.merge_size,
        mean=policy.image_mean,
        std=policy.image_std,
    )


def qwen35_unpatchify(pixel_values: torch.Tensor, grid_thw: torch.Tensor, policy: Qwen35ProcessorPolicy):
    """Inverse of the processor for one image: pixel_values (N, C*tp*p*p) -> image in [0, 1]
    (3, H, W) at the processor-resized size (temporal copy 0)."""
    t, gh, gw = [int(v) for v in grid_thw.reshape(-1).tolist()]
    p, m, tp = policy.patch_size, policy.merge_size, policy.temporal_patch_size
    x = pixel_values.float().reshape(t, gh // m, gw // m, m, m, 3, tp, p, p)[0, ..., 0, :, :]
    # (gh/m, gw/m, m, m, C, p, p) -> (C, gh/m, m, p, gw/m, m, p)
    img = x.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, gh * p, gw * p)
    mean = torch.tensor(policy.image_mean, device=img.device).view(3, 1, 1)
    std = torch.tensor(policy.image_std, device=img.device).view(3, 1, 1)
    return img * std + mean


def _deepstack_indexes(visual: nn.Module) -> list:
    cfg = getattr(visual, "config", None)
    return list(getattr(cfg, "deepstack_visual_indexes", None) or [])


def assert_no_deepstack(visual: nn.Module):
    if _deepstack_indexes(visual) or hasattr(visual, "deepstack_merger_list"):
        raise NotImplementedError(
            "This vision tower has DeepStack taps (Qwen3-VL style): the language model also "
            "receives intermediate visual features, so a single merger-output target is not "
            "sufficient. Only towers with deepstack_visual_indexes = [] (Qwen3.5) are supported."
        )


class Qwen35VisionHead(nn.Module):
    """Qwen3.5 vision blocks [N:] + PatchMerger as the stitched feedforward head.

    stitched features (B, C, h, w) on the 16-px patch grid
        -> merge-window token order (B*h*w, C)
        -> remaining vision blocks (official rot_pos_emb of the tower class, per-image cu_seqlens)
        -> PatchMerger -> (B, h*w/4, out_hidden_size) tokens for the language model
    """

    def __init__(self, blocks, merger, rotary_pos_emb, rot_pos_emb_fn, hidden_size,
                 out_hidden_size, spatial_merge_size, patch_size, num_patches_h, num_patches_w):
        super().__init__()
        self.layers = blocks
        self.merger = merger
        # The official rotary module builds its position range in inv_freq's dtype, so it must
        # keep the tower's dtype (float32) whatever the head is cast to: hold it unregistered
        # (not touched by .to(dtype) / .half()) and only follow the device in forward().
        object.__setattr__(self, "_rotary", rotary_pos_emb)
        self.spatial_merge_size = spatial_merge_size
        self._rot_pos_emb_fn = rot_pos_emb_fn
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.gradient_checkpointing = False
        for param in self.parameters():
            param.requires_grad = False

    @property
    def rotary_pos_emb(self) -> nn.Module:
        """Attribute name used by the official rot_pos_emb (called unbound on this head)."""
        return self._rotary

    @classmethod
    def from_visual(cls, visual: nn.Module, split_layer: int, num_patches_h: int = 32,
                    num_patches_w: int = 32, attn_implementation: str = "sdpa") -> "Qwen35VisionHead":
        """Split a Qwen3.5 vision tower: discard the first `split_layer` blocks, keep the rest.

        num_patches_h/w only set the default target grid; training and evaluation pass the
        grid of each image (target_grid) so the head always runs on the native grid.
        """
        assert_no_deepstack(visual)
        total = len(visual.blocks)
        assert 0 < split_layer <= total, f"split_layer must be in [1, {total}], got {split_layer}"
        kept = copy.deepcopy(nn.ModuleList(list(visual.blocks[split_layer:])))
        for block in kept:
            block.attn.config = copy.copy(block.attn.config)
            block.attn.config._attn_implementation = attn_implementation
        cfg = visual.config
        head = cls(
            blocks=kept,
            merger=copy.deepcopy(visual.merger),
            rotary_pos_emb=copy.deepcopy(visual.rotary_pos_emb),
            rot_pos_emb_fn=type(visual).rot_pos_emb,
            hidden_size=cfg.hidden_size,
            out_hidden_size=cfg.out_hidden_size,
            spatial_merge_size=visual.spatial_merge_size,
            patch_size=cfg.patch_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        logger.info(
            f"Qwen35VisionHead({type(visual).__name__}): keeping blocks [{split_layer}..{total}), "
            f"hidden={cfg.hidden_size}, merger_out={cfg.out_hidden_size}, patch={cfg.patch_size}, "
            f"merge={visual.spatial_merge_size}"
        )
        return head

    def forward(self, stitched_features: torch.Tensor, return_patch_tokens: bool = False, **kwargs):
        x = stitched_features
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, t, h, w = x.shape
        m = self.spatial_merge_size
        hidden = spatial_to_qwen_tokens(x, m).reshape(B * t * h * w, C)
        grid_thw = torch.tensor([[t, h, w]] * B, dtype=torch.long, device=hidden.device)
        if self._rotary.inv_freq.device != hidden.device:
            self._rotary.to(hidden.device)
        rot = self._rot_pos_emb_fn(self, grid_thw).reshape(hidden.shape[0], -1)
        emb = torch.cat((rot, rot), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = qwen2vl_cu_seqlens(grid_thw, hidden.device)
        for block in self.layers:
            if self.gradient_checkpointing and hidden.requires_grad:
                hidden = checkpoint(block, hidden, cu_seqlens=cu_seqlens,
                                    position_embeddings=position_embeddings, use_reentrant=False)
            else:
                hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        merged = self.merger(hidden).reshape(B, t * h * w // (m * m), -1)
        if return_patch_tokens:
            return merged, hidden.reshape(B, t * h * w, C)
        return merged


class _Qwen35Output:
    __slots__ = ("image_embeds", "last_hidden_state", "grid_hw")

    def __init__(self, image_embeds, last_hidden_state, grid_hw):
        self.image_embeds = image_embeds
        self.last_hidden_state = last_hidden_state
        self.grid_hw = grid_hw


class Qwen35TeacherWrapper(nn.Module):
    """Full Qwen3.5 vision tower as the distillation teacher (frames in [0, 1])."""

    def __init__(self, visual: nn.Module, policy: Qwen35ProcessorPolicy):
        super().__init__()
        self.visual = visual
        self.policy = policy

    def forward(self, pixel_values=None, frames=None, **kwargs):
        if frames is None:
            raise ValueError("Qwen35TeacherWrapper needs frames in [0, 1] (frames=...)")
        B = frames.shape[0]
        flat, grid_thw = qwen35_prepare(frames, self.policy)
        out = self.visual(flat.to(self.visual.patch_embed.proj.weight.dtype), grid_thw=grid_thw.to(frames.device))
        merged, hidden = out.pooler_output, out.last_hidden_state
        return _Qwen35Output(
            image_embeds=merged.reshape(B, merged.shape[0] // B, -1),
            last_hidden_state=hidden.reshape(B, hidden.shape[0] // B, -1),
            grid_hw=(int(grid_thw[0, 1]), int(grid_thw[0, 2])),
        )


# ----------------------------------------------------------------------------- loading


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vision_model_class(model_type: str):
    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        return Qwen3_5VisionModel
    raise ValueError(
        f"unsupported model_type {model_type!r}: only dense Qwen3.5 checkpoints (model_type "
        "'qwen3_5', tested with UnifiedReward-2.0-qwen35-{4b,9b,27b}) are supported"
    )


def is_tested_checkpoint(ident: dict) -> bool:
    """True when the identity matches one of TESTED_CHECKPOINTS (revision and config hash)."""
    for repo, entry in TESTED_CHECKPOINTS.items():
        if (ident.get("pinned_revision") == f"{repo}@{entry['revision']}"
                and ident.get("config.json_sha256") == entry["config.json_sha256"]):
            return True
    return False


def _visual_state_dict(checkpoint_dir: str) -> dict:
    """Only the vision-tower tensors (prefix model.visual.), read shard by shard."""
    from safetensors import safe_open

    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        weight_map = json.load(open(index))["weight_map"]
        shards = sorted({f for k, f in weight_map.items() if k.startswith("model.visual.")})
    else:
        shards = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    sd = {}
    for shard in shards:
        with safe_open(os.path.join(checkpoint_dir, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k)
    if not sd:
        raise FileNotFoundError(f"no model.visual.* tensors under {checkpoint_dir}")
    return sd


# "<org>/<name>@<40-hex commit>", as written by scripts/download_weights.py.
_PINNED_REVISION_RE = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*@[0-9a-f]{40}")
# Keys that identify a reward checkpoint; all must be present and equal.
IDENTITY_KEYS = ("pinned_revision", "config.json_sha256", "processor_config.json_sha256")


def _hf_snapshot_revision(checkpoint_dir: str):
    """Revision of a Hugging Face cache snapshot directory
    (.../models--<org>--<name>/snapshots/<40-hex commit>), else None."""
    path = os.path.abspath(checkpoint_dir).rstrip(os.sep)
    commit = os.path.basename(path)
    snapshots = os.path.dirname(path)
    repo_dir = os.path.basename(os.path.dirname(snapshots))
    if (os.path.basename(snapshots) == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", commit)
            and repo_dir.startswith("models--")):
        org, _, name = repo_dir[len("models--"):].partition("--")
        if org and name:
            return f"{org}/{name}@{commit}"
    return None


def checkpoint_identity(checkpoint_dir: str) -> dict:
    """Revision and file hashes identifying a Qwen3.5 checkpoint directory.

    The revision is read from a PINNED_REVISION file ("<repo>@<commit>"), which
    scripts/download_weights.py writes next to the downloaded files, or derived
    from a Hugging Face cache snapshot path. It is None when neither exists;
    require_identity() then refuses the directory.
    """
    rev_file = os.path.join(checkpoint_dir, "PINNED_REVISION")
    if os.path.exists(rev_file):
        with open(rev_file) as f:
            revision = f.read().strip()
    else:
        revision = _hf_snapshot_revision(checkpoint_dir)
    ident = {"checkpoint_dir": os.path.abspath(checkpoint_dir), "pinned_revision": revision}
    for name in ("config.json", "processor_config.json", "preprocessor_config.json"):
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            ident[f"{name}_sha256"] = _sha256(p)
    return ident


def require_identity(ident: dict, what: str = "reward checkpoint"):
    """Refuse an identity without an immutable revision ("<repo>@<40-hex commit>") or without
    the config / processor hashes: such a directory cannot be told apart from another checkpoint
    with the same configuration."""
    revision = ident.get("pinned_revision")
    if not revision or not _PINNED_REVISION_RE.fullmatch(str(revision)):
        raise ValueError(
            f"{what}: no immutable revision (got {revision!r}). Download it with "
            "scripts/download_weights.py (writes PINNED_REVISION = <repo>@<40-hex commit>) or "
            "use a Hugging Face cache snapshot directory"
        )
    missing = [k for k in IDENTITY_KEYS[1:] if not ident.get(k)]
    if missing:
        raise ValueError(f"{what}: missing {missing}")


def assert_same_reward_checkpoint(recorded: dict, current: dict, what: str):
    """Both identities must be complete and equal on IDENTITY_KEYS."""
    require_identity(recorded, f"{what} (recorded)")
    require_identity(current, f"{what} (current)")
    for key in IDENTITY_KEYS:
        if recorded.get(key) != current.get(key):
            raise ValueError(f"{what} belongs to another reward checkpoint ({key}: "
                             f"{recorded.get(key)} != {current.get(key)})")


def load_qwen35_visual(checkpoint_dir: str, device, dtype=torch.bfloat16):
    """Load only the vision tower of a Qwen3.5 checkpoint directory.

    Returns (visual, policy, meta). meta records the revision, file hashes, the vision
    geometry read from the config and the processor policy (stored in Stage-1 sidecars and
    Stage-2 checkpoints so a learned adapter cannot be reused with another checkpoint).
    """
    import transformers
    from transformers import AutoConfig

    require_identity(checkpoint_identity(checkpoint_dir), checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    vcfg = config.vision_config
    vcfg._attn_implementation = "sdpa"
    cls = _vision_model_class(config.model_type)
    visual = cls._from_config(vcfg, torch_dtype=dtype)
    sd = _visual_state_dict(checkpoint_dir)
    visual.load_state_dict(sd, strict=True)
    assert_no_deepstack(visual)
    visual.to(device).eval()
    for p in visual.parameters():
        p.requires_grad = False
    policy = Qwen35ProcessorPolicy.from_checkpoint(checkpoint_dir)
    if (policy.patch_size, policy.merge_size, policy.temporal_patch_size) != (
        vcfg.patch_size, vcfg.spatial_merge_size, vcfg.temporal_patch_size
    ):
        raise ValueError(f"processor {policy} disagrees with vision config {vcfg}")
    meta = {
        **checkpoint_identity(checkpoint_dir),
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "vision_class": cls.__name__,
        "vision": {
            "depth": vcfg.depth, "hidden_size": vcfg.hidden_size, "out_hidden_size": vcfg.out_hidden_size,
            "patch_size": vcfg.patch_size, "spatial_merge_size": vcfg.spatial_merge_size,
            "temporal_patch_size": vcfg.temporal_patch_size,
            "deepstack_visual_indexes": list(getattr(vcfg, "deepstack_visual_indexes", []) or []),
        },
        "visual_tensors": len(sd),
        "processor_policy": asdict(policy),
        "transformers_runtime": transformers.__version__,
        "transformers_checkpoint": getattr(config, "transformers_version", None),
    }
    logger.info(f"Loaded Qwen3.5 vision tower ({len(sd)} tensors, {cls.__name__}) from {checkpoint_dir}")
    if not is_tested_checkpoint(meta):
        logger.warning(
            f"{checkpoint_dir} is not one of the tested UnifiedReward-2.0-qwen35 checkpoints "
            f"(revision {meta.get('pinned_revision')}); results for it have not been validated"
        )
    return visual, policy, meta


def check_search_sidecar(init_path: str, meta: dict, split_layer: int, dit_split: int):
    """Refuse a Stage-1 initialization fitted for another checkpoint or another layer pair."""
    sidecar = os.path.join(os.path.dirname(os.path.abspath(init_path)), "search_meta.json")
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"{sidecar} missing: the Qwen3.5 Stage-1 search writes it next to its init files")
    with open(sidecar) as f:
        found = json.load(f)["reward_model"]
    assert_same_reward_checkpoint(found, meta, f"Stage-1 init {init_path} (searched on another checkpoint?)")
    expected = f"state_dict_dit_blocks_{dit_split}_enc_blocks_{split_layer}_post.pt"
    if os.path.basename(init_path) != expected:
        raise ValueError(f"init file {os.path.basename(init_path)} does not match the location "
                         f"enc_blocks_{split_layer}_dit_{dit_split} (expected {expected})")
