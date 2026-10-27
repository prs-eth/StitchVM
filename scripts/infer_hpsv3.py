#!/usr/bin/env python
"""Score images with the stitched HPSv3 value model (SD3.5 Medium latents -> HPSv3 reward).

Loads a portable (or training) checkpoint plus the pinned upstream models (SD3.5 Medium,
HPSv3 and its Qwen2-VL-7B-Instruct base; see scripts/download_weights.py) and prints one JSON
line per image with the stitched reward at the requested noise level. Rewards are HPSv3's μ.

Example:
  python scripts/infer_hpsv3.py --checkpoint exports/hpsv3_sd3.5_medium/model_trainables_ema.pt \\
    --prompt "a red bicycle leaning against a brick wall" --image a.png b.png --sigma 0.25 --original
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="StitchVM HPSv3 model_trainables_ema.pt")
    ap.add_argument("--image", nargs="+", required=True, help="image files (one prompt for all)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="noise level of the VAE latent, z = (1 - sigma) z0 + sigma eps (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed")
    ap.add_argument("--vae_input_size", type=int, default=1024)
    ap.add_argument("--hpsv3_checkpoint", default=None, help="local HPSv3.safetensors (default: pinned Hub file)")
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--original", action="store_true", help="also score with the original HPSv3 model")
    args = ap.parse_args()

    from stitchvm.models.hpsv3_value_model import StitchedHPSv3

    model = StitchedHPSv3.from_pretrained(args.checkpoint, hpsv3_checkpoint=args.hpsv3_checkpoint,
                                          attn_implementation=args.attn_implementation,
                                          vae_input_size=args.vae_input_size)
    prompts = [args.prompt] * len(args.image)
    stitched = model.score(args.image, prompts, sigma=args.sigma, seed=args.seed)
    original = model.score_original(args.image, prompts) if args.original else [None] * len(args.image)
    for path, s, o in zip(args.image, stitched, original):
        row = {"image": path, "sigma": args.sigma, "seed": args.seed, "stitched_reward": s}
        if args.original:
            row["original_reward"] = o
        print(json.dumps(row))


if __name__ == "__main__":
    main()
