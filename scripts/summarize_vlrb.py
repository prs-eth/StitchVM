"""Validate and summarize saved VL-RewardBench runs: original plus stitched noise levels.

Standard library only: works on saved records.jsonl files (from eval_unifiedreward_vlrb.py,
merged with merge_vlrb_shards.py) without torch or transformers.

Checks, for every run:
  - run provenance (run_meta.json, stitchvm/utils/vlrb_run_meta.py): the original directory
    is an original-mode run and each SIGMA=DIR is a stitched run at exactly that sigma; all runs
    share the reward-checkpoint identity, dataset, item selection and protocol (prompt, decoding,
    image policy, answer order); all stitched runs share the trainables SHA-256 and seed; the run
    covers items 0..n_items-1 and every record carries the run's id;
  - exactly the item indices 0..n_items-1, each once;
  - identical pairing with the original run: same id, expected verdict and answer order
    (swapped) for every index (the benchmark repeats some source ids, so the index is the key);
  - the stored baseline parses equal a fresh parse of the raw output, and both category labels
    equal the mapping of the item id.
Any failure stops the script with exit code 1 before metrics are written.

Legacy inputs: directories written before run_meta.json existed (for example the archived runs
behind the published tables) are refused unless --legacy_inputs is given. With it they are
rescored offline and labelled "legacy" in the output; their own summary.json, when present, is
still checked for mode, sigma, reward identity, dataset hash and decoding. Legacy directories
can never be resumed or merged.

Reported, for every run and every noise level (none is selected):
  - official parser (official UnifiedReward substring rule, stored as official_correct): the
    benchmark-comparable score; also the strict rule (exactly one verdict string), the invalid
    count, per-category accuracy and macro averages under both category mappings;
  - explicit-preference parser (stitchvm.utils.explicit_preference, optional, label-blind,
    applied identically to original and stitched outputs): accuracy (unrecognized outputs count
    as wrong), unrecognized count, reason counts, per-category results;
  - paired comparison of each stitched run with the original for both parsers: correct only in
    one run, exact two-sided McNemar p-value (unadjusted), verdict agreement on items where both
    verdicts are valid, accuracy difference.

Usage:
  python scripts/summarize_vlrb.py --original RUN_DIR \\
      --stitched 0=SIGMA0_DIR 0.1=SIGMA01_DIR ... --output summary.json [--records_out DIR]
      [--legacy_inputs]
"""
import argparse
import json
import os
import statistics
import sys

from stitchvm.utils import explicit_preference as EP
from stitchvm.utils import vlrb_run_meta as RM
from stitchvm.utils.vlrb_protocol import (
    ANSWER_1,
    category_by_source,
    category_official_script,
    mcnemar_exact,
    parse_verdict,
    summarize,
)

MAPPINGS = ("category_official_script", "category_by_source")
CATEGORIES = ("general", "hallucination", "reasoning")


def load_run(run_dir, n_items):
    path = os.path.join(run_dir, "records.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    idx = [r["index"] for r in rows]
    errors = []
    if sorted(idx) != list(range(n_items)):
        missing = sorted(set(range(n_items)) - set(idx))
        dup = sorted({i for i in idx if idx.count(i) > 1})
        errors.append(f"{path}: {len(rows)} records, {len(missing)} missing, {len(dup)} duplicate indices")
    for r in rows:
        p = parse_verdict(r["output"], r["expected"])
        if any(p[k] != r[k] for k in ("official_correct", "strict_verdict", "strict_correct", "invalid")):
            errors.append(f"{path}: index {r['index']}: stored baseline parse differs from the raw output")
            break
        if (r["category_official_script"] != category_official_script(r["id"])
                or r["category_by_source"] != category_by_source(r["id"])):
            errors.append(f"{path}: index {r['index']}: category labels differ from the item id")
            break
    return {r["index"]: r for r in rows}, errors


def _sigma(tag):
    try:
        return float(tag)
    except ValueError:
        return None


def _legacy_view(run_dir):
    """The provenance an archived directory's own summary.json states (None if it has none)."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        s = json.load(f)
    return {"mode": s.get("mode"), "sigma": s.get("sigma"), "seed": s.get("seed"),
            "reward_checkpoint": RM.portable_identity(s.get("reward_checkpoint") or {}),
            "dataset_sha256": s.get("vlrb_parquet_sha256"), "decoding": s.get("decoding")}


def check_provenance(runs, loaded, n_items, legacy_inputs=False):
    """Provenance errors over all runs (see the module docstring) and a per-run provenance label."""
    errors, provenance, views = [], {}, {}
    for tag, path in runs.items():
        meta = RM.read(path)
        want_mode = "original" if tag == "original" else "stitched"
        if meta is None:
            if not legacy_inputs:
                errors.append(f"{path}: no {RM.RUN_META}; archived outputs are rescored only with --legacy_inputs")
                continue
            view = _legacy_view(path)
            provenance[tag] = {"run_meta": False, "legacy": True,
                               "checked_against": "archived summary.json" if view else "nothing (unverified)"}
            if view is None:
                continue
        else:
            rid_ok = RM.allowed_run_ids(meta)
            foreign = [i for i, r in loaded[tag].items() if r.get("run_id") not in rid_ok]
            if foreign:
                errors.append(f"{path}: {len(foreign)} records do not belong to its run (e.g. index {foreign[0]})")
            if meta.get("item_indices") != list(range(n_items)):
                errors.append(f"{path}: run_meta does not cover items 0..{n_items - 1} (a single shard?)")
            view = {"mode": meta.get("mode"), "sigma": meta.get("sigma"), "seed": meta.get("seed"),
                    "reward_checkpoint": meta.get("reward_checkpoint"), "dataset_sha256": meta["dataset"]["sha256"],
                    "decoding": meta.get("decoding"), "trainables_sha256": (meta.get("trainables") or {}).get("sha256"),
                    "protocol": {k: meta.get(k) for k in RM.PROTOCOL_FIELDS if k != "decoding"},
                    "selection": [meta.get("items"), meta.get("n_selected")]}
            provenance[tag] = {"run_meta": True, "legacy": False, "run_id": RM.run_id(meta),
                               "trainables": meta.get("trainables")}
        views[tag] = view
        if view["mode"] != want_mode:
            errors.append(f"{path}: labelled {tag!r} but its run is mode {view['mode']!r}")
        elif tag != "original" and (_sigma(tag) is None or view["sigma"] != _sigma(tag)):
            errors.append(f"{path}: labelled sigma {tag} but its run used sigma {view['sigma']}")
    for key in ("reward_checkpoint", "dataset_sha256", "decoding", "protocol", "selection"):
        vals = {tag: v[key] for tag, v in views.items() if key in v}
        ref_tag = next(iter(vals), None)
        for tag, val in vals.items():
            if val != vals[ref_tag]:
                errors.append(f"{runs[tag]}: {key} differs from {runs[ref_tag]} ({val} != {vals[ref_tag]})")
    for key in ("trainables_sha256", "seed"):
        vals = {tag: v.get(key) for tag, v in views.items() if tag != "original" and key in v}
        ref_tag = next(iter(vals), None)
        for tag, val in vals.items():
            if val != vals[ref_tag]:
                errors.append(f"{runs[tag]}: stitched {key} differs from {runs[ref_tag]} ({val} != {vals[ref_tag]})")
    return errors, provenance


def explicit_metrics(rows):
    a = list(rows.values())
    n = len(a)
    out = {
        "parser_version": EP.VERSION, "n": n,
        "accuracy": sum(r["ep_correct"] for r in a) / n,
        "unrecognized": sum(r["ep"]["verdict"] is None for r in a),
        "newly_recognized_vs_strict": sum(r["strict_verdict"] is None and r["ep"]["verdict"] is not None for r in a),
        "newly_rejected_vs_strict": sum(r["strict_verdict"] is not None and r["ep"]["verdict"] is None for r in a),
        "reason_counts": {},
    }
    for r in a:
        out["reason_counts"][r["ep"]["reason"]] = out["reason_counts"].get(r["ep"]["reason"], 0) + 1
    for mapping in MAPPINGS:
        per = {}
        for c in CATEGORIES:
            sub = [r for r in a if r[mapping] == c]
            per[c] = {"n": len(sub), "accuracy": sum(r["ep_correct"] for r in sub) / max(1, len(sub)),
                      "unrecognized": sum(r["ep"]["verdict"] is None for r in sub)}
        out[mapping] = per
        out[f"macro_{mapping}"] = statistics.mean(v["accuracy"] for v in per.values())
    return out


def paired(orig, run, correct_key, verdict_key):
    b = sum(orig[i][correct_key] and not r[correct_key] for i, r in run.items())
    c = sum(not orig[i][correct_key] and r[correct_key] for i, r in run.items())
    both = [i for i, r in run.items() if verdict_key(r) is not None and verdict_key(orig[i]) is not None]
    n = len(run)
    return {
        "n": n, "original_only_correct": b, "stitched_only_correct": c,
        "mcnemar_exact_p": mcnemar_exact(b, c),
        "delta_accuracy": (sum(r[correct_key] for r in run.values()) - sum(o[correct_key] for o in orig.values())) / n,
        "both_valid": len(both),
        "verdict_agreement": sum(verdict_key(run[i]) == verdict_key(orig[i]) for i in both) / max(1, len(both)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="run directory of the original reward model")
    ap.add_argument("--stitched", nargs="*", default=[], metavar="SIGMA=DIR",
                    help="stitched run directories, one per noise level")
    ap.add_argument("--n_items", type=int, default=1247)
    ap.add_argument("--output", required=True, help="summary JSON")
    ap.add_argument("--records_out", default=None,
                    help="optional directory for per-run records.jsonl with the explicit-preference parse added")
    ap.add_argument("--legacy_inputs", action="store_true",
                    help="also accept directories without run_meta.json (archived outputs; rescoring only)")
    args = ap.parse_args()

    runs = {"original": args.original}
    for spec in args.stitched:
        sigma, _, path = spec.partition("=")
        if not path:
            ap.error(f"--stitched expects SIGMA=DIR, got {spec!r}")
        runs[sigma] = path

    loaded, errors = {}, []
    for tag, path in runs.items():
        rows, errs = load_run(path, args.n_items)
        loaded[tag] = rows
        errors += errs
    prov_errors, provenance = check_provenance(runs, loaded, args.n_items, args.legacy_inputs)
    errors += prov_errors
    orig = loaded["original"]
    for tag, rows in loaded.items():
        for i, r in rows.items():
            o = orig.get(i)
            if o is None or (r["id"], r["expected"], r["swapped"]) != (o["id"], o["expected"], o["swapped"]):
                errors.append(f"{runs[tag]}: index {i} is not paired with the original run (id/expected/order)")
                break
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    for rows in loaded.values():
        for r in rows.values():
            # The parser sees the output text only; the label is used afterwards for scoring.
            r["ep"] = EP.parse(r["output"])
            r["ep_correct"] = r["ep"]["verdict"] == (1 if r["expected"] == ANSWER_1 else 2)

    summary = {
        "n_items": args.n_items, "runs": {}, "paired_vs_original": {},
        "parsers": {
            "official": "official UnifiedReward substring rule (benchmark-comparable)",
            "strict": "exactly one of the two verdict strings",
            "explicit_preference": f"{EP.VERSION}: optional label-blind parser, same rule for all runs",
        },
        "note": "All noise levels are reported; none is selected on this benchmark. "
                "McNemar p-values are unadjusted.",
        "provenance": provenance,
    }
    for tag, rows in loaded.items():
        recs = [rows[i] for i in sorted(rows)]
        summary["runs"][tag] = {"dir": runs[tag], "official": summarize(recs),
                                "explicit_preference": explicit_metrics(rows)}
        if tag != "original":
            summary["paired_vs_original"][tag] = {
                "official": paired(orig, rows, "official_correct", lambda r: r["strict_verdict"]),
                "explicit_preference": paired(orig, rows, "ep_correct", lambda r: r["ep"]["verdict"]),
            }
        if args.records_out:
            dest = os.path.join(args.records_out, tag)
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "records.jsonl"), "w") as f:
                for r in recs:
                    out = {k: v for k, v in r.items() if k not in ("ep", "ep_correct")}
                    out["explicit_preference"] = {"version": EP.VERSION, "verdict": r["ep"]["verdict"],
                                                  "reason": r["ep"]["reason"], "correct": r["ep_correct"]}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"{'run':>9} | {'official':>8} {'invalid':>7} | {'explicit':>8} {'unrecog.':>8} | "
          f"{'p (official)':>12} {'p (explicit)':>12}")
    for tag, s in summary["runs"].items():
        p = summary["paired_vs_original"].get(tag)
        print(f"{tag:>9} | {100 * s['official']['official_accuracy']:8.2f} {s['official']['invalid_parses']:7d} | "
              f"{100 * s['explicit_preference']['accuracy']:8.2f} {s['explicit_preference']['unrecognized']:8d} | "
              + (f"{p['official']['mcnemar_exact_p']:12.3g} {p['explicit_preference']['mcnemar_exact_p']:12.3g}"
                 if p else f"{'—':>12} {'—':>12}"))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
