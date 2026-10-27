"""HPSv3 preference-prediction benchmark for StitchVM (HPDv2 + ImageReward).

Metric: Preference Prediction Accuracy (%), the HPSv3 paper's Table 4 protocol.

Two test sets:
  - hpdv2:         N-way ranking -> inversion-score accuracy (mean over prompts)
  - imagereward:   ranked groups with ties -> ImageReward-style pairwise accuracy
                   (mean over prompts)

The pooled pair accuracy of HPSv3's own evaluate.py (`calc_rank_acc`: every
comparable pair of every prompt, model ties counted wrong) is reported
alongside as `hpsv3_rank_acc`.

Reward model path
-----------------
Original: the full HPSv3 model (Qwen2-VL-7B vision tower + LLM + reward head),
          inputs built exactly like HPSv3RewardInferencer (prompt template,
          256 x 28 x 28 pixel budget), see stitchvm/models/qwen2vl_reward.py.
Stitched: diffusion VAE -> DiT prefix -> stitching layer -> Qwen2-VL vision
          blocks [N:] + merger. The merged tokens replace the vision tower
          output; the language branch, prompt and reward head are the
          original HPSv3 ones. The stitched head is interpolated to each
          image's native HPSv3 patch grid so the language branch sees the same
          token layout as for the original model.

Usage
-----
    python data_processing/prepare_hpdv2_test.py --output_dir data/hpsv3_benchmark/hpdv2
    python data_processing/prepare_imagereward_benchmark.py --output_dir data/hpsv3_benchmark/imagereward

    python scripts/eval_hpsv3_benchmark.py \\
        --checkpoint_path outputs/stitchvm_hpsv3_sd3/checkpoints/epoch_0005/model_trainables.pt \\
        --use_ema --stitching_layer_config conv2d_k1x1_o1280_s1x1_p0x0 \\
        --stitching_layer_location enc_blocks_2_dit_5 \\
        --vae_input_size 1024 --dataset hpdv2 \\
        --data_path data/hpsv3_benchmark/hpdv2 --image_path data/hpsv3_benchmark/hpdv2/test \\
        --noise_levels 0.1 0.25 0.5 0.75 0.9 --eval_original \\
        --output results/hpsv3_sd3_hpdv2.json
"""

import argparse
import json
import os

import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from stitchvm.utils import extension_checkpoint as C
from stitchvm.models.hpsv3_value_model import StitchedHPSv3, check_hpsv3_identity
from stitchvm.models.hpsv3_value_model import build_stitched_hpsv3 as build_stitched_hpsv3_model
from stitchvm.models.qwen2vl_reward import Qwen2VLRewardModel, hpsv3_checkpoint_file


# =============================================================================
# Metrics
# =============================================================================


def inversion_score(p1, p2):
    """1 - (inversions / total_pairs); used for HPDv2 N-way ranking."""
    assert len(p1) == len(p2), f"{len(p1)}, {len(p2)}"
    n = len(p1)
    if n < 2:
        return 1.0
    cnt = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (p1[i] > p1[j] and p2[i] < p2[j]) or (p1[i] < p1[j] and p2[i] > p2[j]):
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)


def calc_image_reward(pred, gt):
    """ImageReward-style pairwise accuracy with tie handling (rank: lower = better)."""
    tol_cnt = 0.0
    true_cnt = 0.0
    for i in range(len(gt)):
        for j in range(i + 1, len(gt)):
            if gt[i] > gt[j]:
                tol_cnt += 1
                if pred[i] < pred[j]:
                    true_cnt += 1
            elif gt[i] < gt[j]:
                tol_cnt += 1
                if pred[i] > pred[j]:
                    true_cnt += 1
    return true_cnt / tol_cnt if tol_cnt else 0.0


def rank_pair_counts(scores, ranking):
    """HPSv3 evaluate.calc_rank_acc counts for one prompt: (correct, total)."""
    true_cnt = tol_cnt = 0
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            if ranking[i] > ranking[j]:
                tol_cnt += 1
                true_cnt += scores[i] < scores[j]
            elif ranking[i] < ranking[j]:
                tol_cnt += 1
                true_cnt += scores[i] > scores[j]
    return true_cnt, tol_cnt


def _rankings_from_scores(scores):
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    return [order.index(j) for j in range(len(scores))]


class _Accumulator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.metric_sum = 0.0
        self.n_groups = 0
        self.pair_true = 0
        self.pair_tot = 0

    def add(self, scores, ranking):
        if self.dataset == "hpdv2":
            self.metric_sum += inversion_score(_rankings_from_scores(scores), ranking)
        else:
            self.metric_sum += calc_image_reward(scores, ranking)
        t, n = rank_pair_counts(scores, ranking)
        self.pair_true += t
        self.pair_tot += n
        self.n_groups += 1

    def result(self):
        acc = self.metric_sum / max(self.n_groups, 1)
        pooled = self.pair_true / max(self.pair_tot, 1)
        return {
            "accuracy": acc, "accuracy_pct": acc * 100,
            "hpsv3_rank_acc_pct": pooled * 100,
            "n_prompts": self.n_groups, "n_pairs": self.pair_tot,
        }


# =============================================================================
# Data
# =============================================================================


def load_groups(dataset, data_path, image_path):
    if dataset == "hpdv2":
        data = json.load(open(os.path.join(data_path, "test.json")))
        return [
            (s["prompt"], [os.path.join(image_path, p) for p in s["image_path"]], s["rank"])
            for s in data
        ]
    data = json.load(open(os.path.join(data_path, "ImageReward_test.json")))
    return [
        (s["prompt"], [os.path.join(image_path, p) for p in s["generations"]], s["ranking"])
        for s in data
    ]


# =============================================================================
# Scoring
# =============================================================================


@torch.no_grad()
def score_original(model, groups, device, acc):
    for prompt, paths, ranking in tqdm(groups, desc="Original HPSv3"):
        inputs, _ = model.build_image_inputs([prompt] * len(paths), paths)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        scores = model.rewards(model(**inputs)).tolist()
        acc.add(scores, ranking)
    return acc.result()


@torch.no_grad()
def score_stitched(value_model, groups, acc, sigma):
    """Stitched rewards per prompt group; noise from the global RNG (seeded per noise level)."""
    model = value_model.reward_model
    for prompt, paths, ranking in tqdm(groups, desc=f"Stitched HPSv3 σ={sigma}"):
        inputs, _ = model.build_image_inputs([prompt] * len(paths), paths)
        grids = inputs.pop("image_grid_thw").tolist()
        inputs.pop("pixel_values")
        images = torch.stack([value_model.vae_transform(Image.open(p).convert("RGB")) for p in paths])
        embeds = value_model.vision_tokens(images, grids, sigma, generator=None)
        acc.add(value_model._reward(inputs, embeds), ranking)
    return acc.result()


# =============================================================================
# Model building
# =============================================================================


def resolve_checkpoint(args):
    """(path actually loaded, checkpoint). --use_ema only selects the path: model_trainables_ema.pt
    next to --checkpoint_path. Whether the loaded weights are EMA is recorded separately from the
    loaded file (stitched_provenance)."""
    ckpt_path = args.checkpoint_path
    if args.use_ema:
        ckpt_path = os.path.join(os.path.dirname(ckpt_path), "model_trainables_ema.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--use_ema set but {ckpt_path} not found.")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    return ckpt_path, C.load_checkpoint(ckpt_path)


def stitched_provenance(args, ckpt_path, ckpt):
    """Result-file record of the stitched model: identity of the file actually loaded (content
    SHA-256, kind, EMA status; no local path) and the --use_ema path-selector flag."""
    return {"trainables": C.trainables_identity(ckpt_path, ckpt),
            "stitching_layer_location": args.stitching_layer_location,
            "use_ema_flag": bool(args.use_ema)}


def build_stitched_hpsv3(args, reward_model, device, resolved=None):
    """Stitched value model from --checkpoint_path (training or portable checkpoint)."""
    ckpt_path, ckpt = resolved or resolve_checkpoint(args)
    saved_loc = ckpt["stitching_layer_location"]
    if saved_loc != args.stitching_layer_location:
        raise ValueError(
            f"Checkpoint was trained at {saved_loc!r} but --stitching_layer_location="
            f"{args.stitching_layer_location!r} was passed."
        )
    out_channels = ckpt["stitching_layer"]["stitching_conv.weight"].shape[0]
    if out_channels != args.stitching_layer_config.out_channels:
        raise ValueError(f"checkpoint adapter width {out_channels} != --stitching_layer_config")
    check_hpsv3_identity(ckpt, args.hpsv3_checkpoint or hpsv3_checkpoint_file(),
                         from_hub_cache=args.hpsv3_checkpoint is None)
    stitched_model, prompt_embedding = build_stitched_hpsv3_model(ckpt, reward_model, device)
    return StitchedHPSv3(reward_model, stitched_model, prompt_embedding, device, args.vae_input_size)


def _default_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--hpsv3_checkpoint", type=str, default=None,
                        help="HPSv3.safetensors (default: MizzenAI/HPSv3 from the Hub)")
    parser.add_argument("--diffusion_model", type=str, default="sd3.5", choices=["sd3.5"])
    parser.add_argument("--stitching_layer_config", type=str, default=None)
    parser.add_argument("--stitching_layer_location", type=str, default=None)
    parser.add_argument("--vae_input_size", type=int, default=1024,
                        help="Square VAE input resolution (release eval convention).")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--use_ema", action="store_true",
                        help="path selector: load model_trainables_ema.pt next to --checkpoint_path. The result "
                             "file records the loaded file's identity (trainables.ema) and this flag (use_ema_flag).")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Reward-model attention (default: flash_attention_2 when "
                             "flash-attn is installed, as the official HPSv3 code does; else sdpa).")
    parser.add_argument("--dataset", type=str, required=True, choices=["hpdv2", "imagereward"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N prompts (smoke tests).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the evaluation noise.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval_original", action="store_true")
    parser.add_argument("--eval_original_only", action="store_true")
    args = parser.parse_args()
    if args.eval_original_only:
        args.eval_original = True
    if args.output is None:
        args.output = f"results/hpsv3_benchmark_{args.dataset}.json"
    device = torch.device("cuda")

    attn = args.attn_implementation or _default_attn()
    reward_model = Qwen2VLRewardModel.load_hpsv3(device, checkpoint_path=args.hpsv3_checkpoint,
                                                 attn_implementation=attn)
    groups = load_groups(args.dataset, args.data_path, args.image_path)
    if args.limit:
        groups = groups[:args.limit]
    metric = "inversion_score" if args.dataset == "hpdv2" else "calc_image_reward"
    results = {"dataset": args.dataset, "metric": metric, "attn_implementation": attn,
               "n_prompts": len(groups)}

    if args.eval_original:
        r = score_original(reward_model, groups, device, _Accumulator(args.dataset))
        results["original_hpsv3"] = r
        logger.info(f"Original HPSv3 — {args.dataset}: {r['accuracy_pct']:.2f}% "
                    f"(HPSv3 rank acc {r['hpsv3_rank_acc_pct']:.2f}%)")

    if not args.eval_original_only:
        assert args.checkpoint_path and args.stitching_layer_config and args.stitching_layer_location
        from stitchvm.models.stitching_layer_builder import parse_conv_spec
        args.stitching_layer_config = parse_conv_spec(args.stitching_layer_config)
        resolved = resolve_checkpoint(args)
        value_model = build_stitched_hpsv3(args, reward_model, device, resolved)
        results["stitched_hpsv3"] = {}
        for k, sigma in enumerate(args.noise_levels):
            torch.manual_seed(args.seed + k)
            r = score_stitched(value_model, groups, _Accumulator(args.dataset), float(sigma))
            r["sigma"] = sigma
            results["stitched_hpsv3"][f"sigma_{sigma:g}"] = r
            logger.info(f"Stitched HPSv3 σ={sigma}: {r['accuracy_pct']:.2f}% "
                        f"(HPSv3 rank acc {r['hpsv3_rank_acc_pct']:.2f}%)")
        results["stitched_hpsv3"].update(stitched_provenance(args, *resolved))

    # Published accuracy of the ORIGINAL HPSv3 model (HPSv3 paper, Table 4): the
    # un-stitched baseline, not a target for the stitched rows.
    results["reference_original"] = {"source": "HPSv3 paper Table 4",
                                     "value": {"hpdv2": 85.4, "imagereward": 66.8}[args.dataset]}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
