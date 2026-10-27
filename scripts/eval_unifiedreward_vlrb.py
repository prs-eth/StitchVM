"""VL-RewardBench generative pairwise-judge evaluation of a UnifiedReward-2.0 Qwen3.5 checkpoint.

Modes:
  original       the reward checkpoint as published
  stitched       vision branch replaced by a Stage-2 StitchVM checkpoint (--trainables)
  split_teacher  identity injection (original blocks [:N] -> head [N:] + merger); with --parity
                 the last-position logits and the greedy generation are compared against the
                 original for every item

Protocol (identical for original and stitched runs): the UnifiedReward pair-rank prompt pinned
in stitchvm/models/qwen35_judge.py, the checkpoint's chat template with enable_thinking=False,
the checkpoint processor applied once to the original image, greedy decoding
(max_new_tokens 4096), and the official answer order (random.seed(0), one draw per item). The
stitched model uses a fixed noise level --sigma and seed --seed. Nothing is trained or tuned on
the benchmark.

Items: --items smoke (12 fixed items: the first 4 of each source category in dataset order),
all, or a comma-separated list of dataset indices. Before generating anything the run writes
<out>/run_meta.json (stitchvm/utils/vlrb_run_meta.py: mode, sigma, seed, reward-checkpoint
identity, SHA-256 and kind of --trainables, dataset hash, item selection and shard, prompt,
decoding; no local paths). It then writes <out>/records.jsonl (one record per item, including the
raw output and the run id) and, once the run finishes, <out>/summary.json.

Each record stores two baseline parses of the output (stitchvm.models.qwen35_judge):
official_correct is the official UnifiedReward substring rule; strict_verdict requires exactly
one verdict string. scripts/summarize_vlrb.py validates complete runs and adds the optional
explicit-preference parser.

Long runs: --num_shards / --shard_index split the items round-robin (rows[i::n]) across
processes; --resume continues an interrupted shard only if its run_meta.json matches the
command (directories without run_meta.json are never resumed); scripts/merge_vlrb_shards.py
validates and merges the shard directories.
"""

import argparse
import hashlib
import io
import json
import os
import time

# Less fragmentation at the largest processor sizes (allocator only; numerics unchanged).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from stitchvm.models import qwen35_judge as J  # noqa: E402
from stitchvm.utils import vlrb_run_meta as RM  # noqa: E402
from stitchvm.utils.vlrb_protocol import summarize  # noqa: E402

# Training reward_meta entries repeated in summary.json (identity and recipe; no local paths).
STITCHED_META_KEYS = RM.IDENTITY_KEYS + ("stitching_layer_location", "stitching_layer_config", "diffusion_model",
                                          "resolution_buckets", "teacher_input", "loss", "ema_decay")


def load_vlrb(parquet: str):
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()
    draws = J.official_order_draws(len(rows), seed=0)
    for i, r in enumerate(rows):
        r["_index"] = i
        r["_draw"] = draws[i]
    return rows


def select(rows, items: str):
    if items == "all":
        return rows
    if items == "smoke":
        out, seen = [], {}
        for r in rows:
            c = J.category_by_source(r["id"])
            if seen.get(c, 0) < 4:
                out.append(r)
                seen[c] = seen.get(c, 0) + 1
        return out
    idx = [int(x) for x in items.split(",")]
    return [rows[i] for i in idx]


def build_stitched(args, judge, device):
    """Stitched vision branch (stitchvm.models.unifiedreward_value_model.build_stitched_visual)."""
    from stitchvm.models.unifiedreward_value_model import build_stitched_visual

    return build_stitched_visual(args.trainables, args.checkpoint, judge, args.sigma, args.seed, device)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_meta(args, reward_identity, trainables, n_rows, selection, rows):
    """Provenance of this run (stitchvm/utils/vlrb_run_meta.py), fixed before generation."""
    import transformers

    stitched = args.mode == "stitched"
    return {
        "format": RM.FORMAT, "version": RM.VERSION, "evaluator": "scripts/eval_unifiedreward_vlrb.py",
        "mode": args.mode, "sigma": float(args.sigma) if stitched else None, "seed": args.seed if stitched else None,
        "split_layer": args.split_layer, "parity": bool(args.parity),
        "reward_checkpoint": RM.portable_identity(reward_identity), "trainables": trainables,
        "dataset": {"name": "VL-RewardBench", "file": os.path.basename(args.vlrb_parquet),
                    "sha256": _sha256(args.vlrb_parquet), "n_rows": n_rows},
        "items": args.items, "n_selected": len(selection), "shard": [args.shard_index, args.num_shards],
        "item_indices": [r["_index"] for r in rows],
        "prompt_source": J.PROMPT_SOURCE, "chat_template_kwargs": {"enable_thinking": False},
        "decoding": {**J.GREEDY, "max_new_tokens": args.max_new_tokens},
        "image_policy": "checkpoint processor (processor_config.json), applied once to the original image",
        "order": "official random.seed(0), one draw per item in dataset order",
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="UnifiedReward-2.0 Qwen3.5 checkpoint directory (scripts/download_weights.py)")
    ap.add_argument("--vlrb_parquet", required=True,
                    help="VL-RewardBench data/test-00000-of-00001.parquet (1,247 items)")
    ap.add_argument("--items", default="smoke")
    ap.add_argument("--mode", choices=["original", "stitched", "split_teacher"], default="original")
    ap.add_argument("--trainables", default=None,
                    help="stitched mode: training or portable model_trainables_ema.pt")
    ap.add_argument("--split_layer", type=int, default=None, help="split_teacher mode")
    ap.add_argument("--parity", action="store_true", help="split_teacher: compare with original per item")
    ap.add_argument("--sigma", type=float, default=0.0, help="stitched mode: fixed noise level")
    ap.add_argument("--seed", type=int, default=0, help="stitched mode: noise seed")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--num_shards", type=int, default=1, help="split the selected items round-robin")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run with identical settings (checked against run_meta.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image

    from stitchvm.utils.extension_checkpoint import trainables_identity
    from stitchvm.models.unifiedreward_value_model import load_judge
    from stitchvm.models.feedforward.qwen35_vision_head import (
        Qwen35VisionHead,
        assert_same_reward_checkpoint,
        checkpoint_identity,
        require_identity,
    )

    if args.mode == "stitched" and not args.trainables:
        ap.error("--mode stitched needs --trainables")
    if args.mode != "stitched" and args.trainables:
        ap.error("--trainables is only used with --mode stitched")
    if args.mode == "split_teacher" and not args.split_layer:
        ap.error("--mode split_teacher needs --split_layer")

    # Provenance first: refuse a mismatched --resume before loading any model.
    reward_identity = checkpoint_identity(args.checkpoint)
    require_identity(reward_identity, args.checkpoint)
    all_dataset_rows = load_vlrb(args.vlrb_parquet)
    selection = select(all_dataset_rows, args.items)
    all_rows = selection[args.shard_index::args.num_shards]
    trainables = None
    if args.mode == "stitched":
        trainables = trainables_identity(args.trainables)
        if trainables["reward_identity"] is None:
            raise SystemExit(f"{args.trainables} has no reward_meta: not a Qwen3.5 StitchVM checkpoint")
        assert_same_reward_checkpoint(trainables["reward_identity"], reward_identity,
                                      f"stitched checkpoint {args.trainables}")
    meta = run_meta(args, reward_identity, trainables, len(all_dataset_rows), selection, all_rows)
    try:
        rid, records = RM.start_or_resume(args.out, meta, args.resume)
    except ValueError as e:
        raise SystemExit(str(e))
    done = {r["index"] for r in records}
    if records:
        print(f"resume: {len(done)} items already in {args.out}/records.jsonl (run {rid[:12]})", flush=True)

    device = torch.device("cuda")
    t0 = time.time()
    judge = load_judge(args.checkpoint, device, max_new_tokens=args.max_new_tokens)
    load_s = time.time() - t0

    visual, extra = None, {}
    if args.mode == "stitched":
        visual, extra = build_stitched(args, judge, device)
    elif args.mode == "split_teacher":
        head = Qwen35VisionHead.from_visual(judge.original_visual, args.split_layer).to(device)
        head.to(judge.original_visual.dtype)
        visual = J.SplitTeacherVisual(judge.original_visual, head, args.split_layer)

    rec_path = os.path.join(args.out, "records.jsonl")
    rows = [r for r in all_rows if r["_index"] not in done]
    torch.cuda.reset_peak_memory_stats()
    for r in rows:
        image = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        r1, r2, expected = J.arrange(r["response"], r["human_ranking"], r["_draw"])
        inputs = judge.inputs(image, J.build_prompt(r["query"], r1, r2))
        t1 = time.time()
        text, n_new = judge.generate(inputs, visual)
        rec = {"index": r["_index"], "id": r["id"], "query_source": r["query_source"],
               "category_official_script": J.category_official_script(r["id"]),
               "category_by_source": J.category_by_source(r["id"]), "expected": expected,
               "swapped": r1 != r["response"][0], "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
               "prompt_tokens": int(inputs["input_ids"].shape[1]), "output": text, "new_tokens": n_new,
               "seconds": time.time() - t1, **J.parse_verdict(text, expected), "run_id": rid}
        if args.mode == "stitched":
            rec["vae_encode_device"] = visual.last_vae_device
        if args.parity and args.mode == "split_teacher":
            ref_text, _ = judge.generate(inputs, None)
            la, lb = judge.last_logits(inputs, None), judge.last_logits(inputs, visual)
            rec.update(parity_logits_maxabs=float((la - lb).abs().max()), parity_generation_equal=ref_text == text,
                       original_output=ref_text)
        records.append(rec)
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("index", "expected", "official_correct", "strict_verdict",
                                               "new_tokens", "seconds")}), flush=True)

    summary = {
        "run_id": rid, "mode": args.mode, "items": args.items, "item_indices": meta["item_indices"],
        "shard": meta["shard"], "reward_checkpoint": meta["reward_checkpoint"], "trainables": trainables,
        "vlrb_parquet": meta["dataset"]["file"], "vlrb_parquet_sha256": meta["dataset"]["sha256"],
        **{k: meta[k] for k in RM.PROTOCOL_FIELDS},
        "sigma": meta["sigma"], "seed": meta["seed"], "split_layer": args.split_layer, "load_seconds": load_s,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "metrics": summarize(records), **{k: v for k, v in extra.items() if k != "reward_meta"},
    }
    if args.parity and args.mode == "split_teacher":
        summary["parity"] = {"max_logits_maxabs": max(r["parity_logits_maxabs"] for r in records),
                             "generations_equal": sum(r["parity_generation_equal"] for r in records),
                             "n": len(records)}
    if "reward_meta" in extra:
        summary["stitched_reward_meta"] = {k: extra["reward_meta"].get(k) for k in STITCHED_META_KEYS}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary["metrics"], indent=1))


if __name__ == "__main__":
    main()
