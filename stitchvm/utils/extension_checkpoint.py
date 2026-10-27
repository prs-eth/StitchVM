"""Portable checkpoints of the HPSv3 and UnifiedReward-2.0 Qwen3.5 extensions.

A portable checkpoint keeps the repository's checkpoint convention — one
``model_trainables_ema.pt`` with ``stitching_layer``, ``stitched_3d_model`` and
``stitching_layer_location`` — so the existing loaders read it unchanged. It holds only the
trained tensors (affine adapter, residual branch, reward-encoder suffix and merger; no frozen
diffusion, VAE, text-encoder or language-model weights, no optimizer state) plus a small
allowlist of JSON-safe metadata:

- ``export_meta``: format version, extension, public model IDs with immutable revisions and file
  hashes, layer pair, head geometry, preprocessing and training recipe;
- ``reward_meta`` (Qwen3.5 only): the reward-checkpoint identity and recipe used by the
  identity guards, without local paths.

Every entry loads with ``torch.load(..., weights_only=True)``.
"""

import json
import os
import re

import torch

FORMAT = "stitchvm-extension-checkpoint"
FORMAT_VERSION = 1
SD35_MEDIUM = {"repo_id": "stabilityai/stable-diffusion-3.5-medium",
               "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80"}
TENSOR_KEYS = ("stitching_layer", "stitched_3d_model")

# reward_meta entries kept in a portable Qwen3.5 checkpoint (identity, geometry and recipe).
REWARD_META_ALLOWLIST = (
    "pinned_revision", "config.json_sha256", "processor_config.json_sha256", "model_type",
    "architectures", "vision_class", "vision", "visual_tensors", "processor_policy",
    "transformers_runtime", "transformers_checkpoint", "stitching_layer_location",
    "stitching_layer_config", "initialization_sha256", "diffusion_model", "resolution_buckets",
    "fixed_resolution", "teacher_input", "loss", "timestep_range", "timestep_distribution",
    "ema_decay",
)
_LOCAL_PATH = re.compile(r"(^|[\s\"'])(/|~/|[A-Za-z]:\\)")


def _json_safe(obj):
    return json.loads(json.dumps(obj))


def find_local_paths(obj, where="metadata"):
    """List string values that look like local filesystem paths (portability check)."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_local_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_local_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str) and _LOCAL_PATH.search(obj):
        hits.append(where)
    return hits


def portable_reward_meta(reward_meta: dict, exclude_list_public: dict | None = None) -> dict:
    """Allowlisted, path-free copy of a training checkpoint's Qwen3.5 reward_meta."""
    meta = {k: reward_meta[k] for k in REWARD_META_ALLOWLIST if k in reward_meta}
    if reward_meta.get("exclude_image_list"):
        # The training run recorded a local list file; keep only the count and point to the
        # public list with the same images.
        meta["exclude_image_list"] = {"n_paths": reward_meta["exclude_image_list"]["n_paths"],
                                      **(exclude_list_public or {})}
    else:
        meta["exclude_image_list"] = None
    return _json_safe(meta)


def build_portable(source: dict, export_meta: dict, reward_meta: dict | None = None) -> dict:
    """Portable checkpoint dict from a training checkpoint dict (tensors copied unchanged)."""
    missing = [k for k in TENSOR_KEYS + ("stitching_layer_location",) if k not in source]
    if missing:
        raise ValueError(f"not a StitchVM training checkpoint (missing {missing})")
    out = {k: {name: t.detach().cpu().contiguous() for name, t in source[k].items()} for k in TENSOR_KEYS}
    out["stitching_layer_location"] = source["stitching_layer_location"]
    out["export_meta"] = _json_safe({"format": FORMAT, "format_version": FORMAT_VERSION, **export_meta})
    if reward_meta is not None:
        out["reward_meta"] = reward_meta
    for key in ("export_meta", "reward_meta"):
        hits = find_local_paths(out.get(key))
        if hits:
            raise ValueError(f"local paths in {key}: {hits}")
    return out


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a training or portable checkpoint with the weights-only unpickler."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    missing = [k for k in TENSOR_KEYS + ("stitching_layer_location",) if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: not a StitchVM checkpoint (missing {missing})")
    meta = ckpt.get("export_meta")
    if meta is not None and (meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION):
        raise ValueError(f"{path}: unsupported checkpoint format {meta.get('format')} v{meta.get('format_version')}")
    return ckpt


def diffusion_backbone(ckpt: dict):
    """Loader arguments for the diffusion backbone of an extension checkpoint: SD3.5 Medium at
    the pinned immutable revision (the only supported backbone). A portable checkpoint must record
    exactly that model and revision; a training checkpoint's reward_meta, if it names a backbone,
    must name sd3.5. Anything else is refused."""
    import argparse

    meta = ckpt.get("export_meta")
    if meta is not None:
        recorded = meta.get("diffusion_model")
        if not isinstance(recorded, dict) or set(recorded) != set(SD35_MEDIUM):
            raise ValueError(f"malformed export_meta.diffusion_model: {recorded!r}")
        if recorded != SD35_MEDIUM:
            raise ValueError(f"checkpoint records diffusion model {recorded['repo_id']}@{recorded['revision']}; "
                             f"only {SD35_MEDIUM['repo_id']}@{SD35_MEDIUM['revision']} is supported")
    reward_backbone = (ckpt.get("reward_meta") or {}).get("diffusion_model", "sd3.5")
    if reward_backbone != "sd3.5":
        raise ValueError(f"checkpoint was trained with diffusion model {reward_backbone!r}; only sd3.5 is supported")
    return argparse.Namespace(diffusion_model="sd3.5", diffusion_revision=SD35_MEDIUM["revision"])


def split_location(location: str) -> tuple[int, int]:
    """'enc_blocks_N_dit_M' -> (N, M)."""
    m = re.fullmatch(r"enc_blocks_(\d+)_dit_(\d+)", location)
    if not m:
        raise ValueError(f"invalid stitching_layer_location {location!r}")
    return int(m.group(1)), int(m.group(2))


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def trainables_identity(path: str, ckpt: dict | None = None) -> dict:
    """Portable identity of a training or portable checkpoint file, as recorded in evaluation
    run metadata: content SHA-256, kind, EMA status, layer pair, export and reward identity."""
    ckpt = ckpt if ckpt is not None else load_checkpoint(path)
    export = ckpt.get("export_meta")
    reward = ckpt.get("reward_meta") or {}
    name = os.path.basename(path)
    return _json_safe({
        "sha256": sha256_file(path),
        "file_name": name,
        "kind": "portable_export" if export else "training_checkpoint",
        # Exports are written from EMA weights only; for training checkpoints the trainer's file
        # name tells EMA (model_trainables_ema.pt) from raw weights (model_trainables.pt).
        "ema": True if export else name == "model_trainables_ema.pt",
        "stitching_layer_location": ckpt["stitching_layer_location"],
        "export": ({k: export.get(k) for k in ("format", "format_version", "extension", "recipe", "source",
                                               "reward_model")} if export else None),
        "reward_identity": ({k: reward.get(k) for k in ("pinned_revision", "config.json_sha256",
                                                        "processor_config.json_sha256")} if reward else None),
    })


def file_sha256_cached(path: str) -> str:
    """SHA-256 of a file; for a Hugging Face cache blob the blob name already is the hash.
    Use only for files the caller obtained from the Hub cache itself (the blob name is a marker
    recorded by the Hub client, not a hash computed here)."""
    real = os.path.realpath(path)
    name = os.path.basename(real)
    if os.sep + "blobs" + os.sep in real and re.fullmatch(r"[0-9a-f]{64}", name):
        return name
    return sha256_file(path)
