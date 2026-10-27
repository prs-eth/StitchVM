#!/usr/bin/env python
"""Export a trained HPSv3 or UnifiedReward-2.0 Qwen3.5 checkpoint in portable form.

Reads a Stage-2 training checkpoint (normally ``.../epoch_0005/model_trainables_ema.pt``) and
writes, to a new directory:

  model_trainables_ema.pt   trained tensors + stitching location + allowlisted metadata
                            (stitchvm/utils/extension_checkpoint.py); weights-only loadable
  stitchvm_export.json      the same metadata plus a per-tensor shape/dtype list
  README.md                 model card (identity, recipe, requirements, licenses)

The source checkpoint is not modified. Tensors are copied unchanged (FP32 EMA weights); no
frozen diffusion, VAE, text-encoder or language-model weights and no optimizer state are
included. Local paths are rejected from the metadata.

Examples:
  python scripts/export_extension_checkpoint.py --recipe stage2_hpsv3 \\
    --checkpoint "$OUT/stitchvm_hpsv3_sd3/checkpoints/epoch_0005/model_trainables_ema.pt" \\
    --output_dir exports/hpsv3_sd3.5_medium
  python scripts/export_extension_checkpoint.py --recipe stage2_unifiedreward_qwen35_9b \\
    --checkpoint "$OUT/stitchvm_unifiedreward_qwen35_9b_sd3/checkpoints/epoch_0005/model_trainables_ema.pt" \\
    --qwen35_checkpoint "$STITCHVM_REWARD_CKPTS/UnifiedReward-2.0-qwen35-9b" \\
    --output_dir exports/unifiedreward_qwen35_9b_sd3.5_medium
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from stitchvm.utils import extension_checkpoint as C  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_LIST = "data_processing/vlrb_hpdv2_near_duplicates.txt"

_COMMON_RECIPE = {
    "data": "HPDv2 train + AVA train",
    "resolution_buckets": [[512, 512], [768, 768], [1024, 1024], [768, 1024], [1024, 768]],
    "epochs": 5, "global_batch": 256, "optimizer": "AdamW", "learning_rate": 5e-5,
    "adapter_lr_multiplier": 1.0, "weight_decay": 0.0, "warmup_updates": 100, "schedule": "constant after warmup",
    "grad_clip_norm": 1.0, "sigma_distribution": "Beta(2,2) on [0, 1]", "ema_decay": 0.999,
    "precision": "bf16 autocast, fp32 trainable copy",
    "adapter": "affine 1x1 conv (Stage-1 least-squares init) + zero-initialized residual branch (width/8, two 1x1 convs)",
    "loss": "MSE over the post-merger image tokens; no language-model or reward-head forward",
}
RECIPES = {
    "stage2_hpsv3": {
        "extension": "hpsv3", "location": "enc_blocks_2_dit_5",
        "reward_model": {"repo_id": "MizzenAI/HPSv3", "revision": "4f81e3e09edd82fe3c5f636444c721b592a735ca",
                         "file": "HPSv3.safetensors"},
        "base_model": {"repo_id": "Qwen/Qwen2-VL-7B-Instruct", "revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac"},
        "teacher_input": "448 x 448 processed training image (32 x 32 patch grid, 256 merged tokens)",
        "evaluation_input": "VAE image resized to 1024 x 1024; stitched head interpolated to the image's native HPSv3 grid",
        "licenses": "HPSv3 weights: Apache-2.0 (Hugging Face model card; code: MIT); Qwen2-VL-7B-Instruct: "
                    "Apache-2.0; Stable Diffusion 3.5 Medium: Stability AI Community License",
    },
    **{f"stage2_unifiedreward_qwen35_{size}": {
        "extension": "unifiedreward_qwen35", "location": loc,
        "reward_model": {"repo_id": f"CodeGoat24/UnifiedReward-2.0-qwen35-{size}", "revision": rev},
        "teacher_input": "processed VAE image at the checkpoint processor's size (native grid = DiT token grid)",
        "evaluation_input": "processor-resized image rebuilt from pixel_values; head on the native grid",
        "exclude_image_list": True,
        "licenses": f"UnifiedReward-2.0-qwen35-{size}: MIT (base Qwen/Qwen3.5-{size.upper()}: Apache-2.0); "
                    "Stable Diffusion 3.5 Medium: Stability AI Community License",
    } for size, loc, rev in (
        ("4b", "enc_blocks_2_dit_4", "a1cb85ca3dd2c65ff561cc1d8fd66e6830c5ab30"),
        ("9b", "enc_blocks_2_dit_3", "f01548b009741e12ff9817ed91dba94701ed9579"),
        ("27b", "enc_blocks_2_dit_3", "3a5c836f485047a96f45754e08c981ca9e41fa60"),
    )},
}


def _head_geometry(ckpt):
    head = ckpt["stitched_3d_model"]
    blocks = sorted({int(m.group(1)) for k in head for m in [re.match(r"layers\.(\d+)\.", k)] if m})
    conv = ckpt["stitching_layer"]["stitching_conv.weight"]
    merger_out = [v for k, v in head.items() if k.startswith("merger.") and k.endswith(".weight") and v.ndim == 2]
    return {"suffix_blocks": len(blocks), "stitching_conv_in_out": [int(conv.shape[1]), int(conv.shape[0])],
            "kernel_size": list(conv.shape[2:]), "merger_out": int(merger_out[-1].shape[0])}


def _model_card(meta, name):
    rm = meta["reward_model"]
    loc = meta["stitching_layer_location"]
    n, m = C.split_location(loc)
    rec = meta["training_recipe"]
    geo = meta["head_geometry"]
    hpsv3 = meta["extension"] == "hpsv3"
    guide = "docs/EXTENSION_HPSV3.md" if hpsv3 else "docs/EXTENSION_UNIFIEDREWARD_QWEN35.md"
    cli = ("python scripts/infer_hpsv3.py --checkpoint model_trainables_ema.pt --prompt \"...\" --image a.png --sigma 0"
           if hpsv3 else
           "python scripts/infer_unifiedreward_judge.py --checkpoint model_trainables_ema.pt "
           f"--reward_checkpoint \"reward_ckpts/{rm['repo_id'].split('/')[-1]}\" --image photo.jpg --question \"...\" "
           "--answer1 \"...\" --answer2 \"...\" --sigma 0")
    buckets = ", ".join(f"{w}×{h}" for w, h in rec["resolution_buckets"])
    lines = [
        f"# StitchVM {name}",
        "",
        f"StitchVM value-model weights for **{meta['diffusion_model']['repo_id']}** stitched into the vision branch of "
        f"**{rm['repo_id']}**. The file contains only the trained adapter and reward-encoder suffix (vision blocks "
        "after the split plus the merger); the diffusion model, VAE, text encoders and the reward model's language "
        "branch are loaded from their original sources.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Format | `{meta['format']}` v{meta['format_version']} (`model_trainables_ema.pt`, loads with "
        "`torch.load(..., weights_only=True)`) |",
        f"| Weights | {meta['source']['weights']}, epoch {meta['source']['epoch']} |",
        f"| Reward model | `{rm['repo_id']}` @ `{rm['revision']}` |",
    ]
    if rm.get("file_sha256"):
        lines.append(f"| Reward model file | `{rm['file']}`, SHA-256 `{rm['file_sha256']}` |")
    if meta.get("base_model"):
        lines.append(f"| Reward base model | `{meta['base_model']['repo_id']}` @ `{meta['base_model']['revision']}` |")
    if rm.get("config.json_sha256"):
        lines.append(f"| Reward config / processor SHA-256 | `{rm['config.json_sha256']}` / "
                     f"`{rm['processor_config.json_sha256']}` |")
    lines += [
        f"| Diffusion backbone | `{meta['diffusion_model']['repo_id']}` @ `{meta['diffusion_model']['revision']}` |",
        f"| Layer pair | DiT block output {m} → reward-encoder output {n} (`{loc}`) |",
        f"| Trained head | adapter {geo['stitching_conv_in_out'][0]} → {geo['stitching_conv_in_out'][1]} channels "
        f"(1×1), {geo['suffix_blocks']} vision blocks, merger to {geo['merger_out']} |",
        f"| Teacher input (training) | {meta['teacher_input']} |",
        f"| Evaluation input | {meta['evaluation_input']} |",
        f"| Recipe | {rec['data']}; VAE buckets {buckets}; {rec['epochs']} epochs; global batch "
        f"{rec['global_batch']}; AdamW, LR {rec['learning_rate']}, {rec['warmup_updates']} warmup updates; "
        f"σ ~ {rec['sigma_distribution']}; EMA {rec['ema_decay']} |",
        f"| Loss | {rec['loss']} |",
    ]
    if meta.get("exclude_image_list"):
        e = meta["exclude_image_list"]
        lines.append(f"| Excluded training images | {e['n_paths']} HPDv2 near-duplicates of VL-RewardBench test images "
                     f"(`{e['public_list']}`) |")
    lines += [
        "",
        "## Usage",
        "",
        f"Follow `{guide}` in the StitchVM repository: install the repository, download the upstream models "
        "with `scripts/download_weights.py`, then:",
        "",
        "```bash",
        cli,
        "```",
        "",
        f"Measured results of these weights and the evaluation protocol are in `{guide}` (Results).",
        "",
        "## Licenses",
        "",
        f"{meta['licenses']}. The trained suffix is a fine-tuned copy of the reward model's vision blocks, so the "
        "reward model's license applies to it; using the weights also requires Stable Diffusion 3.5 Medium under "
        "its license. StitchVM code: Apache-2.0.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--checkpoint", required=True, help="Stage-2 training checkpoint (model_trainables_ema.pt)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--qwen35_checkpoint", default=None,
                    help="UnifiedReward-2.0 Qwen3.5 checkpoint directory (identity check; Qwen3.5 recipes)")
    ap.add_argument("--hpsv3_checkpoint", default=None,
                    help="HPSv3.safetensors used for training (default: pinned Hub file); hashed into the metadata")
    ap.add_argument("--epoch", type=int, default=None, help="Default: parsed from .../epoch_NNNN/ in --checkpoint")
    args = ap.parse_args()

    recipe = RECIPES[args.recipe]
    if os.path.exists(os.path.join(args.output_dir, "model_trainables_ema.pt")):
        raise SystemExit(f"{args.output_dir} already contains an export; use a new directory")
    if os.path.basename(args.checkpoint) != "model_trainables_ema.pt":
        raise SystemExit("export the EMA weights (model_trainables_ema.pt)")
    epoch = args.epoch
    if epoch is None:
        m = re.search(r"epoch_(\d+)", args.checkpoint)
        epoch = int(m.group(1)) if m else None
    src = C.load_checkpoint(args.checkpoint)
    if src["stitching_layer_location"] != recipe["location"]:
        raise SystemExit(f"checkpoint location {src['stitching_layer_location']} != recipe {recipe['location']}")

    export_meta = {
        "extension": recipe["extension"], "recipe": args.recipe,
        "stitching_layer_location": src["stitching_layer_location"],
        "diffusion_model": C.SD35_MEDIUM, "reward_model": dict(recipe["reward_model"]),
        "head_geometry": _head_geometry(src), "teacher_input": recipe["teacher_input"],
        "evaluation_input": recipe["evaluation_input"], "training_recipe": _COMMON_RECIPE,
        "source": {"weights": "EMA (decay 0.999) of the trainable parameters", "epoch": epoch},
        "licenses": recipe["licenses"],
    }
    reward_meta = None
    if recipe["extension"] == "hpsv3":
        from stitchvm.models import qwen2vl_reward as R

        export_meta["base_model"] = recipe["base_model"]
        path = args.hpsv3_checkpoint or R.hpsv3_checkpoint_file()
        # The pinned Hub download is content-addressed in the cache; a user-given file is hashed.
        export_meta["reward_model"]["file_sha256"] = (C.sha256_file(path) if args.hpsv3_checkpoint
                                                      else C.file_sha256_cached(path))
    else:
        from stitchvm.models.feedforward.qwen35_vision_head import assert_same_reward_checkpoint, checkpoint_identity

        if not args.qwen35_checkpoint:
            raise SystemExit("--qwen35_checkpoint is required for the Qwen3.5 recipes")
        source_meta = src.get("reward_meta")
        if source_meta is None:
            raise SystemExit("the training checkpoint has no reward_meta")
        assert_same_reward_checkpoint(source_meta, checkpoint_identity(args.qwen35_checkpoint), "training checkpoint")
        want = f"{recipe['reward_model']['repo_id']}@{recipe['reward_model']['revision']}"
        if source_meta["pinned_revision"] != want:
            raise SystemExit(f"checkpoint was trained on {source_meta['pinned_revision']}, recipe expects {want}")
        public_list = {"public_list": EXCLUDE_LIST,
                       "public_list_sha256": C.sha256_file(os.path.join(REPO_ROOT, EXCLUDE_LIST))}
        reward_meta = C.portable_reward_meta(source_meta, public_list)
        export_meta["reward_model"].update({k: source_meta[k] for k in ("config.json_sha256",
                                                                        "processor_config.json_sha256")})
        export_meta["exclude_image_list"] = reward_meta["exclude_image_list"]

    portable = C.build_portable(src, export_meta, reward_meta)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "model_trainables_ema.pt")
    torch.save(portable, out)

    # Verify: weights-only reload, same keys, bitwise-equal tensors.
    back = C.load_checkpoint(out)
    for key in C.TENSOR_KEYS:
        if set(back[key]) != set(src[key]):
            raise SystemExit(f"{key}: tensor names differ after export")
        for name, t in src[key].items():
            if back[key][name].dtype != t.dtype or not torch.equal(back[key][name], t.cpu()):
                raise SystemExit(f"{key}.{name}: tensor differs after export")
    tensors = {f"{k}.{n}": {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
               for k in C.TENSOR_KEYS for n, t in back[k].items()}
    with open(os.path.join(args.output_dir, "stitchvm_export.json"), "w") as f:
        json.dump({"export_meta": back["export_meta"], "reward_meta": back.get("reward_meta"),
                   "num_tensors": len(tensors),
                   "num_parameters": int(sum(t.numel() for k in C.TENSOR_KEYS for t in back[k].values())),
                   "tensors": tensors}, f, indent=1)
    name = os.path.basename(os.path.normpath(args.output_dir))
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(_model_card(back["export_meta"], name))
    print(json.dumps({"output": out, "num_tensors": len(tensors), "verified_bitwise_equal": True}))


if __name__ == "__main__":
    main()
