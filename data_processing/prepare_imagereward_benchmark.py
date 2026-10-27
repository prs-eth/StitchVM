"""Download and prepare the ImageReward test set (preference prediction benchmark).

The ImageRewardDB dataset is hosted at https://huggingface.co/datasets/THUDM/ImageRewardDB
(pinned to revision c493721e19e296eb615420036f2a2eed08412bb4). The test split has 2,720 images
across 466 prompts with (possibly tied) ranking labels.

This script reads the test split straight from the Hub files
(``metadata-test.parquet`` + ``images/test/test_{1,2}.zip``) — no `datasets`
script loader needed — and writes the format the HPSv3 benchmark reads:

    output_dir/
        ImageReward_test.json   <- [{prompt, generations: [file, ...], ranking: [int, ...]}, ...]
        images/                 <- test images

Prompts are sorted by prompt_id and the images of a prompt by rank (stable for
ties), and files are named ``{prompt_id}_{i:02d}_r{rank}.<ext>``.

``--format original`` (default) extracts the original WebP files byte-for-byte —
the unmodified inputs, used for the primary results. ``--format jpeg95``
re-encodes every image as JPEG quality 95, a lossy variant of the inputs; its
results are reported separately as a compatibility number.

Requires pandas and pyarrow (optional evaluation dependencies).

Usage:
    python data_processing/prepare_imagereward_benchmark.py --output_dir data/hpsv3_benchmark/imagereward
    python data_processing/prepare_imagereward_benchmark.py --output_dir data/hpsv3_benchmark/imagereward_jpeg95 --format jpeg95
"""

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict

from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
from tqdm import tqdm

REPO_ID = "THUDM/ImageRewardDB"
# Dataset revision the HPSv3 extension results were measured on.
REVISION = "c493721e19e296eb615420036f2a2eed08412bb4"


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ImageReward test set")
    parser.add_argument("--output_dir", type=str, default="data/hpsv3_benchmark/imagereward")
    parser.add_argument("--format", choices=["original", "jpeg95"], default="original")
    args = parser.parse_args()

    import pandas as pd

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    meta_file = os.path.join(args.output_dir, "ImageReward_test.json")
    if os.path.exists(meta_file):
        logger.info(f"{meta_file} already exists — delete it to re-create. Skipping.")
        return

    meta = pd.read_parquet(hf_hub_download(REPO_ID, "metadata-test.parquet", repo_type="dataset", revision=REVISION))
    logger.info(f"Loaded {len(meta)} test entries")
    zips = {}
    for i in (1, 2):
        zf = zipfile.ZipFile(hf_hub_download(REPO_ID, f"images/test/test_{i}.zip", repo_type="dataset",
                                             revision=REVISION))
        for name in zf.namelist():
            zips[name] = zf

    groups = defaultdict(list)
    for row in meta.itertuples(index=False):
        groups[row.prompt_id].append(row)
    logger.info(f"Found {len(groups)} unique prompts")

    test_data = []
    for prompt_id, rows in tqdm(sorted(groups.items()), desc="Processing prompts"):
        rows = sorted(rows, key=lambda r: r.rank)  # stable: ties keep file order
        generations, rankings = [], []
        for i, row in enumerate(rows):
            src = os.path.basename(row.image_path)
            data = zips[src].read(src)
            ext = "jpg" if args.format == "jpeg95" else src.rsplit(".", 1)[-1]
            # The enumerate index is load-bearing: ImageRewardDB has TIED ranks,
            # so a filename keyed on rank alone would collide within a prompt.
            filename = f"{prompt_id}_{i:02d}_r{row.rank}.{ext}"
            path = os.path.join(image_dir, filename)
            if not os.path.exists(path):
                if args.format == "jpeg95":
                    Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=95)
                else:
                    with open(path, "wb") as f:
                        f.write(data)
            generations.append(filename)
            rankings.append(int(row.rank))
        test_data.append({"prompt": rows[0].prompt, "generations": generations, "ranking": rankings})

    with open(meta_file, "w") as f:
        json.dump(test_data, f)
    logger.info(
        f"ImageReward test set ready: {len(test_data)} prompts, "
        f"{sum(len(s['generations']) for s in test_data)} images ({args.format}) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
