"""Validate and merge eval_unifiedreward_vlrb.py shard directories (standard library only).

Shard directories are named <prefix>_shard<i>of<n> (round-robin split of the selected items,
rows[i::n]). The merge reads each shard's run_meta.json and records.jsonl (a shard's summary.json
is optional: a shard that was interrupted and resumed still has every record) and checks:
  - every shard has run_meta.json (stitchvm/utils/vlrb_run_meta.py); directories written
    without it are not merged;
  - the run metadata of all shards agree apart from the shard fields (mode, sigma, seed, reward
    checkpoint, trainables SHA-256, dataset, item selection, prompt, decoding, ...), and each
    shard's metadata names its own shard and item indices;
  - each shard holds exactly its expected item indices (no missing, extra or duplicate items) and
    every record carries its shard's run id;
  - the union is exactly range(--n_items) (all 1,247 VL-RewardBench items for --items all).
On any violation nothing is merged: <out>/INCOMPLETE.json lists the problems and the script
exits 1. Rerun the incomplete shards with --resume (only missing items run), then merge again.
The merged directory gets its own run_meta.json (shard fields replaced by the merged item list
and the shard run ids), so summarize_vlrb.py can verify it.

Usage:
  python scripts/merge_vlrb_shards.py OUT_DIR SHARD_DIR [SHARD_DIR ...] [--n_items 1247]
"""
import argparse
import json
import os
import re

from stitchvm.utils import vlrb_run_meta as RM
from stitchvm.utils.vlrb_protocol import summarize


def merged_meta(common: dict, shard_ids: list, n_items: int) -> dict:
    return {**common, "shard": None, "item_indices": list(range(n_items)), "merged_from": shard_ids}


def merge(out, shards, n_items=1247):
    report, records, metas, shard_ids, summaries = {"shards": {}}, [], [], [], []
    expected_all = list(range(n_items))
    ok = True
    for d in shards:
        m = re.search(r"_shard(\d+)of(\d+)$", d.rstrip("/"))
        if not m:
            raise ValueError(f"{d}: expected a <prefix>_shard<i>of<n> directory")
        i, n = int(m.group(1)), int(m.group(2))
        if n != len(shards):
            raise ValueError(f"{d}: shard count {n} != {len(shards)} directories given")
        expected = expected_all[i::n]
        meta = RM.read(d)
        rs = RM.read_records(d)
        idx = [r["index"] for r in rs]
        problems = []
        if meta is None:
            problems.append(f"no {RM.RUN_META} (written without run provenance; not merged)")
        else:
            if meta.get("shard") != [i, n]:
                problems.append(f"run_meta shard {meta.get('shard')} != directory name [{i}, {n}]")
            if meta.get("item_indices") != expected:
                problems.append("run_meta item_indices differ from this shard's items")
            if meta.get("n_selected") != n_items:
                problems.append(f"run_meta n_selected {meta.get('n_selected')} != --n_items {n_items}")
            rid = RM.run_id(meta)
            foreign = sum(r.get("run_id") != rid for r in rs)
            if foreign:
                problems.append(f"{foreign} records do not carry this shard's run id")
            metas.append(meta)
            shard_ids.append({"shard": [i, n], "run_id": rid})
        missing = sorted(set(expected) - set(idx))
        extra = sorted(set(idx) - set(expected))
        dup = sorted({x for x in idx if idx.count(x) > 1})
        summary_path = os.path.join(d, "summary.json")
        report["shards"][d] = {"expected": len(expected), "records": len(rs), "missing": missing,
                               "extra": extra, "duplicates": dup, "problems": problems,
                               "summary_json": os.path.exists(summary_path)}
        ok &= not (missing or extra or dup or problems)
        records += rs
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summaries.append(json.load(f))
    union = sorted(r["index"] for r in records)
    report["union_equals_expected"] = union == expected_all
    ok &= report["union_equals_expected"]
    for mt in metas[1:]:
        diff = RM.differences(metas[0], mt, ignore=RM.SHARD_FIELDS)
        if diff:
            report.setdefault("meta_mismatch", []).extend(k for k in diff if k not in report.get("meta_mismatch", []))
            ok = False
    stitched_metas = [sm.get("stitched_reward_meta") for sm in summaries]
    if any(sm != stitched_metas[0] for sm in stitched_metas[1:]):
        report.setdefault("meta_mismatch", []).append("stitched_reward_meta")
        ok = False
    os.makedirs(out, exist_ok=True)
    if not ok:
        with open(os.path.join(out, "INCOMPLETE.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({d: {k: (v if not isinstance(v, list) else len(v)) for k, v in s.items()}
                          for d, s in report["shards"].items()}, indent=1))
        print(f"INCOMPLETE: not merged ({out}/INCOMPLETE.json)")
        return 1
    common = {k: v for k, v in metas[0].items() if k not in RM.SHARD_FIELDS}
    meta = RM.normalized(merged_meta(common, shard_ids, n_items))
    old = RM.read(out)
    if old is not None and RM.differences(old, meta):
        raise ValueError(f"{out} already holds a different merged run ({RM.RUN_META}); use a new output directory")
    if os.path.exists(os.path.join(out, "INCOMPLETE.json")):
        os.remove(os.path.join(out, "INCOMPLETE.json"))
    rid = RM.run_id(meta) if old is not None else RM.write(out, meta)
    records.sort(key=lambda r: r["index"])
    with open(os.path.join(out, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    stitched = stitched_metas[0] if stitched_metas else None
    summary = {"run_id": rid, "mode": meta["mode"], "reward_checkpoint": meta["reward_checkpoint"],
               "trainables": meta["trainables"], "vlrb_parquet_sha256": meta["dataset"]["sha256"],
               "sigma": meta["sigma"], "seed": meta["seed"], **{k: meta[k] for k in RM.PROTOCOL_FIELDS},
               "shards": shards, "n_items": len(records), "validation": report,
               "shard_summaries_present": len(summaries),
               "peak_memory_gb": max((sm.get("peak_memory_gb") or 0) for sm in summaries) if summaries else None,
               "vae_encode_device_counts": {d: sum(1 for r in records if r.get("vae_encode_device") == d)
                                            for d in sorted({r.get("vae_encode_device") for r in records}, key=str)},
               "metrics": summarize(records)}
    if stitched is not None:
        summary["stitched_reward_meta"] = stitched
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary["metrics"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--n_items", type=int, default=1247)
    args = ap.parse_args()
    raise SystemExit(merge(args.out, args.shards, args.n_items))


if __name__ == "__main__":
    main()
