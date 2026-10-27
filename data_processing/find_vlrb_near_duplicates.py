"""Find training images that are near-duplicates of VL-RewardBench test images.

Used to build data_processing/vlrb_hpdv2_near_duplicates.txt, the exclusion list of the
UnifiedReward-2.0 Qwen3.5 Stage-2 recipes. Only the benchmark *images* are read; no labels,
questions or answers are used.

Training lists are built with the training dataset classes themselves (HPDv2Dataset image list,
AVATrainingDataset split="train"), so exactly the files training reads are checked. For every image:
  phash  64-bit DCT hash of a 32x32 grayscale thumbnail (8x8 low frequencies vs their median)
  dhash  64-bit horizontal gradient hash of a 9x8 grayscale thumbnail
Candidates: Hamming(phash) <= --max_hamming or Hamming(dhash) <= --max_hamming against any test image
(also against the horizontally flipped test image). Each candidate is verified with the Pearson
correlation of 64x64 grayscale thumbnails; confirmed duplicates (corr >= --min_corr) are written to
the exclusion list as paths relative to their dataset root (the format --exclude_image_list reads).

Usage:
  python data_processing/find_vlrb_near_duplicates.py \
    --vlrb_parquet "$DATA/vl_rewardbench/data/test-00000-of-00001.parquet" \
    --hpdv2_root "$DATA/hpdv2" --ava_root "$DATA/aesthetic_benchmark/ava" \
    --out "$DATA/vlrb_dedup"
Writes <out>/exclude_paths.txt, <out>/candidates.json and <out>/summary.json. Requires pyarrow.
"""

import argparse
import io
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageOps

_DCT = None


def _dct_matrix(n=32):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] /= np.sqrt(2)
    return m * np.sqrt(2 / n)


def _bits(b):
    return int(np.packbits(b.astype(np.uint8).reshape(-1)).view(">u8")[0])


def hashes(img):
    global _DCT
    if _DCT is None:
        _DCT = _dct_matrix()
    g = img.convert("L")
    a32 = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = _DCT @ a32 @ _DCT.T
    low = d[:8, :8].reshape(-1)
    ph = _bits(low > np.median(low[1:]))
    a98 = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    dh = _bits(a98[:, 1:] > a98[:, :-1])
    thumb = np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.float32)
    return ph, dh, thumb


def hash_path(path):
    try:
        img = Image.open(path)
        img.draft("RGB", (128, 128))
        ph, dh, _ = hashes(img)
        return path, ph, dh
    except Exception as e:  # noqa: BLE001
        return path, None, str(e)


def thumb_path(path):
    img = Image.open(path)
    img.draft("RGB", (256, 256))
    return hashes(img)[2]


def popcount64(x):
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8):
        c += _POP8[((x >> np.uint64(s)) & np.uint64(255)).astype(np.uint8)]
    return c


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlrb_parquet", required=True)
    ap.add_argument("--hpdv2_root", required=True)
    ap.add_argument("--ava_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max_hamming", type=int, default=10)
    ap.add_argument("--min_corr", type=float, default=0.95)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    import pyarrow.parquet as pq

    from stitchvm.data.ava_dataset import AVATrainingDataset
    from stitchvm.data.hpdv2_datasets import HPDv2Dataset

    rows = pq.read_table(a.vlrb_parquet, columns=["id", "image"]).to_pylist()
    test = []
    for r in rows:
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        for flip in (False, True):
            ph, dh, th = hashes(ImageOps.mirror(img) if flip else img)
            test.append((r["id"], flip, ph, dh, th))
    tph = np.array([t[2] for t in test], dtype=np.uint64)
    tdh = np.array([t[3] for t in test], dtype=np.uint64)

    hpd = HPDv2Dataset(a.hpdv2_root, 512, 512, 512, 512, color_augmentation=False).image_paths
    ava = AVATrainingDataset(a.ava_root, 512, 512, 512, 512, split="train", color_augmentation=False).samples
    paths = [("HPDv2", p) for p in hpd] + [("ava", p) for p in ava]
    roots = {"HPDv2": os.path.abspath(a.hpdv2_root), "ava": os.path.abspath(a.ava_root)}

    def rel(src, p):
        return os.path.relpath(os.path.abspath(p), roots[src])
    print(f"training images: HPDv2 {len(hpd)}, AVA train {len(ava)}; test images {len(rows)}", flush=True)

    with Pool(a.workers) as pool:
        res = pool.map(hash_path, [p for _, p in paths], chunksize=256)
    failed = [(rel(src, p), err.replace(p, rel(src, p)))
              for (src, _), (p, ph, err) in zip(paths, res) if ph is None]
    ok = [(src, p, ph, dh) for (src, _), (p, ph, dh) in zip(paths, res) if ph is not None]
    print(f"hashed {len(ok)} ({len(failed)} unreadable) in {time.time() - t0:.0f}s", flush=True)

    gph = np.array([o[2] for o in ok], dtype=np.uint64)
    gdh = np.array([o[3] for o in ok], dtype=np.uint64)
    cands = []
    for j in range(len(test)):
        hp = popcount64(gph ^ tph[j])
        hd = popcount64(gdh ^ tdh[j])
        for i in np.nonzero((hp <= a.max_hamming) | (hd <= a.max_hamming))[0]:
            cands.append((int(i), j, int(hp[i]), int(hd[i])))
    print(f"{len(cands)} hash candidates", flush=True)

    out, exclude = [], set()
    for i, j, hp, hd in cands:
        src, p = ok[i][0], ok[i][1]
        c = corr(thumb_path(p), test[j][4])
        dup = c >= a.min_corr
        out.append({"train_source": src, "train_path": rel(src, p), "test_id": test[j][0], "test_flipped": test[j][1],
                    "phash_hamming": hp, "dhash_hamming": hd, "thumb_corr": c, "duplicate": dup})
        if dup:
            exclude.add(rel(src, p))
    json.dump(out, open(os.path.join(a.out, "candidates.json"), "w"), indent=1)
    with open(os.path.join(a.out, "exclude_paths.txt"), "w") as f:
        for p in sorted(exclude):
            f.write(p + "\n")
    summary = {"vlrb_parquet": os.path.basename(a.vlrb_parquet), "test_images": len(rows), "train_hpdv2": len(hpd),
               "train_ava": len(ava), "unreadable": len(failed), "unreadable_examples": failed[:5],
               "max_hamming": a.max_hamming, "min_corr": a.min_corr, "hash_candidates": len(cands),
               "confirmed_duplicates": len(exclude),
               "duplicates_by_source": {s: len({o["train_path"] for o in out if o["duplicate"] and o["train_source"] == s})
                                        for s in ("HPDv2", "ava")},
               "seconds": time.time() - t0}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
