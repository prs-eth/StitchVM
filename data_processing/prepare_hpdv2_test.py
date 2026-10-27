"""Download and prepare the HPDv2 test set (preference-prediction benchmark).

The HPDv2 dataset is hosted at https://huggingface.co/datasets/ymhao/HPDv2 (pinned to
revision bb0f2563e451e11c3f376b2ba93615db00e042e1):
  - test.json:   metadata with {prompt, image_path, rank} per prompt
  - test.tar.gz: the test images referenced by test.json

Usage:
    python data_processing/prepare_hpdv2_test.py --output_dir data/hpsv3_benchmark/hpdv2

Output:
    output_dir/
        test.json   <- metadata
        test/       <- extracted images

Evaluate with scripts/eval_hpsv3_benchmark.py --dataset hpdv2
--data_path <output_dir> --image_path <output_dir>/test.
"""

import argparse
import json
import os
import tarfile

from huggingface_hub import hf_hub_download
from loguru import logger

REPO_ID = "ymhao/HPDv2"
REPO_TYPE = "dataset"
# Dataset revision the HPSv3 extension results were measured on (its test.json has git blob
# b1c6ee8c57ff760b9ed4956f1005aff40fc602d9).
REVISION = "bb0f2563e451e11c3f376b2ba93615db00e042e1"


def main():
    parser = argparse.ArgumentParser(description="Download and prepare the HPDv2 test set")
    parser.add_argument("--output_dir", type=str, default="data/hpsv3_benchmark/hpdv2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    test_json_path = os.path.join(args.output_dir, "test.json")
    test_images_dir = os.path.join(args.output_dir, "test")

    if os.path.exists(test_json_path):
        logger.info(f"{test_json_path} exists, skipping download")
    else:
        hf_hub_download(repo_id=REPO_ID, filename="test.json", repo_type=REPO_TYPE, revision=REVISION,
                        local_dir=args.output_dir)

    if os.path.isdir(test_images_dir) and os.listdir(test_images_dir):
        logger.info(f"{test_images_dir} exists, skipping download")
    else:
        tar_path = hf_hub_download(repo_id=REPO_ID, filename="test.tar.gz", repo_type=REPO_TYPE,
                                   revision=REVISION, local_dir=args.output_dir)
        logger.info(f"Extracting {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=args.output_dir)
        os.remove(tar_path)

    with open(test_json_path) as f:
        test_data = json.load(f)
    missing = [
        p for s in test_data for p in s["image_path"]
        if not os.path.exists(os.path.join(test_images_dir, p))
    ]
    n_images = sum(len(s["image_path"]) for s in test_data)
    logger.info(f"HPDv2 test set: {len(test_data)} prompts, {n_images} images in {test_images_dir}")
    if missing:
        raise SystemExit(f"{len(missing)} images listed in test.json are missing, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
