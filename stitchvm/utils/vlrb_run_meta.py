"""Provenance of VL-RewardBench evaluation runs (standard library only).

scripts/eval_unifiedreward_vlrb.py writes <out>/run_meta.json once, before the first record.
It records what was evaluated (mode, sigma, seed, reward-checkpoint identity, SHA-256 and kind of
the stitched trainables), on which items (dataset hash, item selection, shard) and how (prompt,
decoding, image policy, answer order). It contains no local paths. The SHA-256 of its canonical
JSON is the run id, and every record of the run carries it.

- --resume continues a directory only if its run_meta.json equals the new run's metadata and all
  its records carry that run id.
- scripts/merge_vlrb_shards.py requires run_meta.json in every shard, with identical metadata
  apart from the shard fields.
- scripts/summarize_vlrb.py checks each directory's mode and sigma against its label, and checks
  that all runs share the reward checkpoint, dataset and protocol.

Directories written before run_meta.json existed are legacy outputs. They are never resumed or
merged; only summarize_vlrb.py --legacy_inputs rescores them, and it labels them as such.
"""

import hashlib
import json
import os

RUN_META = "run_meta.json"
FORMAT = "stitchvm-vlrb-run"
VERSION = 1
# Same keys as stitchvm.models.feedforward.qwen35_vision_head.IDENTITY_KEYS (kept torch-free here).
IDENTITY_KEYS = ("pinned_revision", "config.json_sha256", "processor_config.json_sha256")
# Fields that differ between the shards of one run (and in a merged run).
SHARD_FIELDS = ("shard", "item_indices", "merged_from")
# Fields every run compared in one summary must share.
PROTOCOL_FIELDS = ("prompt_source", "chat_template_kwargs", "decoding", "image_policy", "order")


def canonical(meta: dict) -> str:
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def run_id(meta: dict) -> str:
    return hashlib.sha256(canonical(meta).encode()).hexdigest()


def normalized(meta: dict) -> dict:
    """The metadata as it reads back from JSON (tuples become lists)."""
    return json.loads(canonical(meta))


def portable_identity(ident: dict) -> dict:
    """Reward-checkpoint identity without the local directory."""
    return {k: ident.get(k) for k in IDENTITY_KEYS}


def read(run_dir: str):
    """run_meta.json of a run directory, or None for a legacy directory."""
    path = os.path.join(run_dir, RUN_META)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError(f"{path}: not a {FORMAT} v{VERSION} file")
    return meta


def write(run_dir: str, meta: dict) -> str:
    """Write run_meta.json once (refuses to replace one) and make it read-only; returns the run id."""
    path = os.path.join(run_dir, RUN_META)
    with open(path, "x") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(path, 0o444)
    return run_id(meta)


def differences(a: dict, b: dict, ignore=()) -> list:
    return sorted(k for k in set(a) | set(b) if k not in ignore and a.get(k) != b.get(k))


def read_records(run_dir: str) -> list:
    path = os.path.join(run_dir, "records.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def allowed_run_ids(meta: dict) -> set:
    """Run ids a directory's records may carry: its own, or those of the shards it was merged from."""
    return {run_id(meta)} | {m["run_id"] for m in meta.get("merged_from") or []}


def start_or_resume(run_dir: str, meta: dict, resume: bool):
    """Open a run directory before any generation. Returns (run_id, records already written).

    A new run writes run_meta.json first. --resume requires the directory's run_meta.json to
    equal ``meta`` and every existing record to carry its run id; a directory with records but no
    run_meta.json (a legacy output) is never resumed."""
    meta = normalized(meta)
    os.makedirs(run_dir, exist_ok=True)
    old = read(run_dir)
    records = read_records(run_dir)
    if not resume:
        if old is not None or records:
            raise ValueError(f"{run_dir} already holds a run: use --resume to continue it, or a new --out")
        return write(run_dir, meta), []
    if old is None:
        if records:
            raise ValueError(f"{run_dir}: records without {RUN_META} (written without run provenance); "
                             "it cannot be resumed, start a new --out")
        return write(run_dir, meta), []
    diff = differences(old, meta)
    if diff:
        raise ValueError(f"{run_dir}: --resume with settings that differ from {RUN_META}: {diff}")
    rid = run_id(old)
    foreign = [r.get("index") for r in records if r.get("run_id") != rid]
    if foreign:
        raise ValueError(f"{run_dir}: {len(foreign)} records do not belong to this run (e.g. index {foreign[0]})")
    return rid, records
