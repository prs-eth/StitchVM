"""Download and extract the HPDv2 train images used for Stage 1.

Usage:
    python data_processing/prepare_hpdv2_train.py --output_dir data/hpdv2

The output contains train/ (images) and train.json (metadata). Stage 1 reads
images directly, so --skip_json can omit the metadata download. Pass the
output directory as --dataset "HPDv2:/path/to/hpdv2".
"""

import argparse
import os

from huggingface_hub import hf_hub_download
from loguru import logger


REPO_ID = "ymhao/HPDv2"
REPO_TYPE = "dataset"


def _count_images(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        1
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download HPDv2 train set (~430K images, 29.5 GB) for Stage 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hpdv2",
        help=(
            "Output directory. Point at scratch / external storage if your "
            "home quota is small — the extracted train/ dir is ~32 GB."
        ),
    )
    parser.add_argument(
        "--keep_tar",
        action="store_true",
        help=(
            "Keep the downloaded train.tar.gz after extracting. Default is "
            "to remove it (saves 29.5 GB once images are unpacked)."
        ),
    )
    parser.add_argument(
        "--skip_json",
        action="store_true",
        help="Skip downloading train.json (only needed if you want metadata).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_json_path = os.path.join(args.output_dir, "train.json")
    train_images_dir = os.path.join(args.output_dir, "train")

    # Step 1: train.json (metadata, 110 MB)
    if args.skip_json:
        logger.info("--skip_json: not downloading train.json")
    elif os.path.exists(train_json_path):
        logger.info(f"train.json already present at {train_json_path}; skipping")
    else:
        logger.info(f"Downloading train.json (~110 MB) from {REPO_ID}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename="train.json",
            repo_type=REPO_TYPE,
            local_dir=args.output_dir,
        )
        logger.info(f"  saved to {downloaded}")

    # Step 2: train.tar.gz (29.5 GB) + extract
    n_existing = _count_images(train_images_dir)
    if n_existing > 400_000:  # HPDv2 has ~430K unique train images
        logger.info(
            f"train/ already has {n_existing:,} images at {train_images_dir}; "
            f"skipping download + extract"
        )
    else:
        logger.warning(
            "About to download train.tar.gz (29.5 GB). "
            "Make sure you have at least ~62 GB free on the destination "
            "filesystem (29.5 GB tar + 32 GB extracted) before continuing."
        )
        logger.info(f"Downloading train.tar.gz from {REPO_ID}...")
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="train.tar.gz",
            repo_type=REPO_TYPE,
            local_dir=args.output_dir,
        )
        logger.info(f"  saved to {tar_path}")

        logger.info("Extracting train.tar.gz (this takes a few minutes)...")
        import subprocess
        try:
            # Use native tar with GCS-Fuse safe flags:
            # --no-same-permissions: Apply user umask instead of restoring archive permissions
            # --no-same-owner: Do not attempt to restore ownership
            # -m: Do not restore file modification times (GCS Fuse might reject this)
            subprocess.run([
                "tar", "-xf", tar_path,
                "--no-same-permissions",
                "--no-same-owner",
                "-m",
                "-C", args.output_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Native tar extraction failed: {e}")
            raise
        logger.info(f"  extracted to {train_images_dir}")

        if args.keep_tar:
            logger.info(f"--keep_tar set; leaving {tar_path} on disk")
        else:
            os.remove(tar_path)
            logger.info(f"Removed {tar_path} (pass --keep_tar to retain it)")

    # Step 3: verify
    n_final = _count_images(train_images_dir)
    logger.info("HPDv2 train set ready:")
    logger.info(f"  Images dir:   {train_images_dir}")
    logger.info(f"  Image count:  {n_final:,}")
    if not args.skip_json:
        logger.info(f"  Metadata:     {train_json_path}")
    if n_final < 400_000:
        logger.warning(
            f"  Expected ~430K images, found only {n_final:,}. "
            "Extraction may have been interrupted — re-run this script."
        )

    logger.info(
        f"\nUse with Stage 1:\n"
        f"  python find_stitching_layer.py \\\n"
        f"    --dataset HPDv2:{os.path.abspath(args.output_dir)} \\\n"
        f"    ..."
    )


if __name__ == "__main__":
    main()
