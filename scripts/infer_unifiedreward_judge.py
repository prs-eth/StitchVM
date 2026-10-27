#!/usr/bin/env python
"""Pairwise judgement with the stitched UnifiedReward-2.0 Qwen3.5 judge.

Given one image, a question and two candidate answers, the judge generates its rationale and
verdict ("Answer 1 is better" / "Answer 2 is better") with the stitched vision branch at a fixed
noise level (or with the original vision tower, --original_only). This is a pairwise judge, not a
general visual question-answering interface. Prints one JSON object with the generated text and
two parses: the official verdict phrases and the explicit-preference parser.

Example:
  python scripts/infer_unifiedreward_judge.py \\
    --checkpoint exports/unifiedreward_qwen35_9b_sd3.5_medium/model_trainables_ema.pt \\
    --reward_checkpoint "$STITCHVM_REWARD_CKPTS/UnifiedReward-2.0-qwen35-9b" \\
    --image photo.jpg --question "How many dogs are in the picture?" \\
    --answer1 "There are two dogs." --answer2 "There is one cat." --sigma 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="StitchVM Qwen3.5 model_trainables_ema.pt")
    ap.add_argument("--reward_checkpoint", required=True,
                    help="UnifiedReward-2.0 Qwen3.5 directory with PINNED_REVISION (scripts/download_weights.py)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer1", required=True)
    ap.add_argument("--answer2", required=True)
    ap.add_argument("--sigma", type=float, default=0.0, help="noise level of the stitched branch (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--original_only", action="store_true", help="run the original judge instead")
    args = ap.parse_args()
    if not args.original_only and not args.checkpoint:
        ap.error("--checkpoint is required unless --original_only is set")

    from stitchvm.models.unifiedreward_value_model import StitchedUnifiedRewardJudge

    judge = StitchedUnifiedRewardJudge.from_pretrained(
        None if args.original_only else args.checkpoint, args.reward_checkpoint, sigma=args.sigma,
        seed=args.seed, max_new_tokens=args.max_new_tokens)
    print(json.dumps(judge.judge_pair(args.image, args.question, args.answer1, args.answer2), indent=1))


if __name__ == "__main__":
    main()
