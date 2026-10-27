#!/usr/bin/env python
"""Score images with any released StitchVM recipe; see examples/README.md."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("hpsv2", "clip", "aesthetic", "dfn_clip", "hpsv3",
          "unifiedreward_4b", "unifiedreward_9b", "unifiedreward_27b")
BACKBONES = ("sd3.5", "sd3.5_large", "flux_dev")
HUB_REPO = "HJGO/stitchvm"


def checkpoint_path(model, backbone):
    """Published Hub layout; model selection also covers older exports without layer metadata."""
    if model == "hpsv3":
        return "hpsv3_sd3.5_medium/model_trainables_ema.pt"
    if model.startswith("unifiedreward_"):
        size = model.split("_")[-1]
        return f"unifiedreward_qwen35_{size}_sd3.5_medium/model_trainables_ema.pt"
    if backbone == "sd3.5" and model in ("hpsv2", "clip"):
        return f"{model}_sd3.5_medium/model_trainables_ema.pt"
    suffix = {"sd3.5": "sd3", "sd3.5_large": "sd3_large", "flux_dev": "flux"}[backbone]
    return f"stitchvm_{model}_{suffix}/checkpoints/epoch_0005/model_trainables_ema.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list supported combinations without loading models")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--backbone", choices=BACKBONES, default="sd3.5")
    parser.add_argument("--image", nargs="+", help="one or more image paths")
    parser.add_argument("--prompt", help="text to score against; unused for Aesthetic")
    parser.add_argument("--sigma", type=float, default=0.0, help="latent noise level in [0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", help="local StitchVM checkpoint; otherwise downloaded from HJGO/stitchvm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-input-size", type=int, default=None,
                        help="square VAE input for scalar rewards (default 1024); UnifiedReward uses its native grid")
    parser.add_argument("--reward-checkpoint", help="local HPSv3 file or UnifiedReward upstream model directory")
    parser.add_argument("--question", help="UnifiedReward question")
    parser.add_argument("--answer1", help="UnifiedReward first candidate answer")
    parser.add_argument("--answer2", help="UnifiedReward second candidate answer")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="UnifiedReward generation limit")
    args = parser.parse_args(argv)
    if args.list:
        return args
    if not args.model or not args.image:
        parser.error("--model and --image are required (or use --list)")
    if not 0 <= args.sigma <= 1:
        parser.error("--sigma must be in [0, 1]")
    is_judge = args.model.startswith("unifiedreward_")
    if (is_judge or args.model == "hpsv3") and args.backbone != "sd3.5":
        parser.error("HPSv3 and UnifiedReward support --backbone sd3.5 only")
    if is_judge:
        if not all((args.question, args.answer1, args.answer2)):
            parser.error("UnifiedReward requires --question, --answer1 and --answer2")
        if args.vae_input_size is not None:
            parser.error("UnifiedReward uses its processor-native image size; omit --vae-input-size")
        if args.max_new_tokens <= 0:
            parser.error("--max-new-tokens must be positive")
    elif args.model != "aesthetic" and not args.prompt:
        parser.error("--prompt is required for this reward model")
    if args.vae_input_size is not None and (args.vae_input_size <= 0 or args.vae_input_size % 16):
        parser.error("--vae-input-size must be a positive multiple of 16")
    for path in args.image + ([args.checkpoint] if args.checkpoint else []):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for model in MODELS:
            backbones = ("sd3.5",) if model == "hpsv3" or model.startswith("unifiedreward_") else BACKBONES
            for backbone in backbones:
                print(f"{model:20s} {backbone:12s} {checkpoint_path(model, backbone)}")
        return
    reward_root = Path(os.environ.get("STITCHVM_REWARD_CKPTS", ROOT / "reward_ckpts"))
    if args.model.startswith("unifiedreward_"):
        size = args.model.split("_")[-1]
        reward_path = args.reward_checkpoint or str(reward_root / f"UnifiedReward-2.0-qwen35-{size}")
        if not Path(reward_path).is_dir():
            raise FileNotFoundError(f"Run python scripts/download_weights.py unifiedreward_qwen35_{size} first, "
                                    "or provide --reward-checkpoint")
        from stitchvm.models.feedforward.qwen35_vision_head import checkpoint_identity, require_identity
        identity = checkpoint_identity(reward_path)
        require_identity(identity)
        if identity["pinned_revision"].split("@")[0] != f"CodeGoat24/UnifiedReward-2.0-qwen35-{size}":
            raise ValueError("--reward-checkpoint must match the size selected by --model")
    from huggingface_hub import hf_hub_download
    checkpoint = args.checkpoint or hf_hub_download(HUB_REPO, checkpoint_path(args.model, args.backbone))
    if args.model == "hpsv3":
        from stitchvm.models.hpsv3_value_model import StitchedHPSv3
        model = StitchedHPSv3.from_pretrained(
            checkpoint, device=args.device, hpsv3_checkpoint=args.reward_checkpoint,
            vae_input_size=args.vae_input_size or 1024)
        score_type = "hpsv3_reward"
        # Process one image at a time to bound memory and make the seed independent of list order.
        outputs = ({"score": model.score([image], [args.prompt], sigma=args.sigma, seed=args.seed)[0],
                    "score_type": score_type} for image in args.image)
    elif args.model.startswith("unifiedreward_"):
        from stitchvm.models.unifiedreward_value_model import StitchedUnifiedRewardJudge
        model = StitchedUnifiedRewardJudge.from_pretrained(
            checkpoint, reward_path, sigma=args.sigma, seed=args.seed,
            device=args.device, max_new_tokens=args.max_new_tokens)
        outputs = (model.judge_pair(image, args.question, args.answer1, args.answer2) for image in args.image)
    else:
        from stitchvm.models.image_value_model import StitchedImageReward
        model = StitchedImageReward.from_pretrained(
            checkpoint, model=args.model, backbone=args.backbone, device=args.device,
            vae_input_size=args.vae_input_size or 1024,
            hpsv2_checkpoint_dir=str(reward_root / "hpsv2"),
            aesthetic_mlp_path=str(reward_root / "sac+logos+ava1-l14-linearMSE.pth"))
        score_type = "aesthetic_score" if args.model == "aesthetic" else "cosine_similarity"
        outputs = ({"score": model.score([image], args.prompt, args.sigma, args.seed)[0],
                    "score_type": score_type} for image in args.image)
    for image, output in zip(args.image, outputs):
        print(json.dumps({"image": image, "model": args.model, "backbone": args.backbone,
                          "sigma": args.sigma, "seed": args.seed, **output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
